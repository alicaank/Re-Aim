"""Read extracted DL3DV Nerfstudio scenes and build registered camera batches."""

import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .datasets import safe_name
from .geometry import poses


def scenes(path):
    path = Path(path).resolve()
    if path.is_file() and path.name == "transforms.json":
        return [path]
    if (path / "transforms.json").is_file():
        return [path / "transforms.json"]
    found = sorted(path.rglob("transforms.json"))
    if not found:
        raise ValueError(f"No extracted DL3DV transforms.json found under {path}")
    return found


def scene_frames(source, count):
    spec = json.loads(source.read_text())
    frames = sorted(
        spec["frames"],
        key=lambda f: [
            int(part) if part.isdigit() else part
            for part in re.split(r"(\d+)", Path(f["file_path"]).name)
        ],
    )
    if len(frames) < count:
        raise ValueError(
            f"{source.parent.name} has {len(frames)} registered frames. This model needs {count}."
        )
    # Choose consecutive frames. DL3DV may omit poses for failed registrations.
    names = [Path(f["file_path"]).stem for f in frames]
    numbers = [re.search(r"(\d+)$", name) for name in names]
    if not all(numbers):
        raise ValueError(f"Cannot establish frame order in {source}")
    ids = [int(m.group()) for m in numbers]
    start = 0
    for i, num in enumerate(ids):
        if i and num != ids[i - 1] + 1:
            start = i
        if i - start + 1 == count:
            frames = frames[start : i + 1]
            break
    else:
        raise ValueError(
            f"No sequence of {count} consecutive registered frames in {source}"
        )
    first = Path(frames[0]["file_path"])
    directories = [source.parent / first.parent, source.parent / "images"]
    directories += sorted(source.parent.glob("images_*"))
    directory = next((d for d in directories if (d / first.name).is_file()), None)
    if directory is None:
        raise ValueError(f"Cannot locate {first.name} beside {source}")
    paths = [directory / Path(f["file_path"]).name for f in frames]
    missing = next((p for p in paths if not p.is_file()), None)
    if missing:
        raise ValueError(f"Missing recording frame {missing}")
    return spec, frames, paths


def rectification(spec, frames, size):
    fields = ("fl_x", "fl_y", "cx", "cy", "w", "h", "k1", "k2", "p1", "p2", "k3", "k4")

    def parameters(frame):
        values = {**spec, **frame}
        model = values.get("camera_model", "OPENCV")
        if model not in {"OPENCV", "PINHOLE", "SIMPLE_PINHOLE"}:
            raise ValueError(f"Unsupported DL3DV camera model {model}")
        return [
            float(values[k] if i < 6 else values.get(k, 0))
            for i, k in enumerate(fields)
        ]

    intr = parameters(frames[0])
    if any(not np.allclose(parameters(f), intr) for f in frames[1:]):
        raise ValueError("This generator requires fixed camera intrinsics")
    fx, fy, cx, cy, w, h, *distortion = intr
    if min(fx, fy, w, h) <= 0 or not np.isfinite(intr).all():
        raise ValueError("Invalid DL3DV intrinsics")
    sx, sy = size[0] / w, size[1] / h
    source_K = np.array([[fx * sx, 0, cx * sx], [0, fy * sy, cy * sy], [0, 0, 1.0]])
    K = np.array([[fx * 864 / w, 0, 432], [0, fy * 480 / h, 240], [0, 0, 1.0]])
    maps = cv2.initUndistortRectifyMap(
        source_K, np.array(distortion + [0.0, 0.0]), None, K, (864, 480), cv2.CV_32FC1
    )
    return K, maps


def metric_scale(images, w2c, K, metric_depth):
    """Match single-view metric z-depth to triangulated points in the dataset's units."""
    anchor = cv2.imread(str(images[0]), cv2.IMREAD_GRAYSCALE)
    corners = cv2.goodFeaturesToTrack(anchor, 2400, 0.005, 5)
    if corners is None:
        raise ValueError("Cannot calibrate DL3DV scale: no trackable anchor features")
    samples = []
    for frame in sorted(set(min(n, len(images) - 1) for n in (2, 4, 8, 16, 32))):
        target = cv2.imread(str(images[frame]), cv2.IMREAD_GRAYSCALE)
        tracked, ok, _ = cv2.calcOpticalFlowPyrLK(anchor, target, corners, None)
        back, ok_back, _ = cv2.calcOpticalFlowPyrLK(target, anchor, tracked, None)
        keep = (
            (ok.ravel() != 0)
            & (ok_back.ravel() != 0)
            & (np.linalg.norm((back - corners)[:, 0], axis=1) < 1.0)
        )
        a, b = corners[keep, 0].T, tracked[keep, 0].T
        if a.shape[1] < 16:
            continue
        P = w2c[frame, :3]
        pts = cv2.triangulatePoints(K @ w2c[0, :3], K @ P, a, b)
        with np.errstate(divide="ignore", invalid="ignore"):
            xyz = pts[:3] / pts[3]
            moved = P[:, :3] @ xyz + P[:, 3:4]
            pixel = K @ moved
            reproj = np.linalg.norm(pixel[:2] / pixel[2] - b, axis=0)
            ray0 = np.linalg.solve(K, np.vstack((a, np.ones(a.shape[1]))))
            ray1 = P[:, :3].T @ np.linalg.solve(K, np.vstack((b, np.ones(b.shape[1]))))
            sine = np.linalg.norm(np.cross(ray0.T, ray1.T), axis=1) / (
                np.linalg.norm(ray0, axis=0) * np.linalg.norm(ray1, axis=0)
            )
            x, y = np.rint(a).astype(int)
            ratios = metric_depth[y, x] / xyz[2]
        good = (
            (xyz[2] > 0)
            & (moved[2] > 0)
            & (reproj < 1.0)
            & (sine > 0.01)
            & (ratios > 0)
            & np.isfinite(ratios)
        )
        samples.extend(ratios[good])
    if len(samples) < 32:
        raise ValueError(
            "Insufficient triangulation for metric scale. Provide calibrated geometry for this scene."
        )
    lo, hi = np.quantile(samples, [0.1, 0.9])
    if hi / lo > 3:
        raise ValueError("Inconsistent DL3DV metric scale. Supply a calibrated scene.")
    return float(np.median(samples))


def prepare_dl3dv(files, count, dest, observer, depth_worker, depth_checkpoint):
    dest.mkdir(parents=True)
    rows = []
    seen = set()
    for source in files:
        name = safe_name(source.parent.name)
        if name in seen:
            raise ValueError(f"Duplicate scene name {name} under dataset root")
        seen.add(name)
        spec, frames, paths = scene_frames(source, count)
        folder = dest / name
        recording = folder / "recording"
        recording.mkdir(parents=True)
        with Image.open(paths[0]) as image:
            size = image.size
        K, maps = rectification(spec, frames, size)
        images = []
        for i, path in enumerate(paths):
            img = cv2.imread(str(path))
            if img is None or img.shape[1::-1] != size:
                raise ValueError(f"Image grid changed or image unreadable: {path}")
            new = recording / f"frame_{i + 1:05d}.jpg"
            if not cv2.imwrite(
                str(new),
                cv2.remap(img, *maps, cv2.INTER_LINEAR),
                [cv2.IMWRITE_JPEG_QUALITY, 97],
            ):
                raise OSError(f"Cannot write {new}")
            images.append(new)
        gl = poses(np.array([x["transform_matrix"] for x in frames]))
        c2w = gl @ np.diag([1.0, -1.0, -1.0, 1.0])
        w2c = np.linalg.inv(c2w) @ c2w[0]
        np.save(folder / "K.npy", K)
        subprocess.run(
            [
                observer,
                depth_worker,
                images[0],
                folder / "K.npy",
                depth_checkpoint,
                folder / "metric-depth.npy",
            ],
            check=True,
        )
        metric_depth = np.load(folder / "metric-depth.npy", allow_pickle=False)
        if metric_depth.shape != (480, 864):
            raise ValueError("DA3 returned depth on an unexpected pixel grid")
        scale = metric_scale(images, w2c, K, metric_depth)
        np.savez_compressed(
            folder / "geometry.npz",
            w2c=w2c,
            intrinsics=K,
            anchor_depth=metric_depth / scale,
        )
        caption = (
            spec.get("caption")
            or "A video of the scene following the camera trajectory."
        )
        rows.append(
            dict(
                name=name,
                image=str(images[0]),
                frames_dir=str(recording),
                original_start=0,
                caption=caption,
                prompts={"worldplay": caption},
                geometry=str(folder / "geometry.npz"),
                metres_per_anchor_unit=scale,
                metric_scale_evidence="DA3 metric depth / triangulated anchor z-depth",
            )
        )
    manifest = dest / "scenes.json"
    manifest.write_text(
        json.dumps(
            dict(
                dataset="DL3DV",
                split="user-supplied",
                scenes=rows,
                pose_convention="opencv_w2c_anchor_relative",
                depth_convention="z_depth_same_units_as_translation",
            )
        )
    )
    return manifest
