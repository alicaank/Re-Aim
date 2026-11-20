"""VIPE camera evaluation and unchanged DINO/LPIPS image metrics; separate provenance."""

import argparse, json, re, sys, time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

R = Path(__file__).resolve().parent
A = R.parent
sys.path[:0] = [
    str(A / "fair_common_20261025"),
    str(A / "new_models_loop_20261028"),
    "@WORKSPACE@/training_runs/joint_samplerfix_20261012",
    str(A / "memorizon_plugins"),
    "@WORKSPACE@/training_runs/memory1500_resplat_20260905/python_deps",
]
import fair_common as fc
from camera import estimate, errors, path_metrics


def load_frames(source, h, scene):
    source = Path(source)
    if source.suffix == ".npz" or (source / "frames.npz").is_file():
        with np.load(
            source if source.is_file() else source / "frames.npz", allow_pickle=False
        ) as z:
            frames = z["frames"][: h + 1].copy()
    elif (source / f"{scene}_frames").exists():
        frames = np.stack(
            [
                np.asarray(
                    Image.open(
                        source / f"{scene}_frames" / f"frame_{i:05d}.png"
                    ).convert("RGB")
                )
                for i in range(h + 1)
            ]
        )
    else:
        import cv2

        path = source if source.is_file() else source / "review_full.mp4"
        cap = cv2.VideoCapture(str(path))
        images = []
        if not cap.isOpened():
            raise ValueError("Cannot open " + str(path))
        try:
            for i in range(h + 1):
                ok, im = cap.read()
                if not ok:
                    raise ValueError(f"{path}: missing frame {i}")
                images.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        frames = np.stack(images)
    if (
        len(frames) != h + 1
        or frames.dtype != np.uint8
        or frames.ndim != 4
        or frames.shape[-1] != 3
    ):
        raise ValueError("Incomplete or invalid RGB sequence")
    return frames


def output_K(model, K, source_wh, out_wh):
    sw, sh = source_wh
    w, h = out_wh
    if model == "worldplay":
        return fc.k_worldplay(K, source_wh, out_wh)[0]
    if model == "memorizon":
        return fc.k_memorizon(K, source_wh, out_wh)[0]
    if model == "matrixgame":
        cw, ch = min(sw, w), min(sh, h)
        k = K.copy()
        k[0, 2] -= (sw - cw) // 2
        k[1, 2] -= (sh - ch) // 2
        return fc.scale_crop_K(k, w / cw, h / ch)
    if model == "vmem":
        return fc.k_vmem(K, source_wh, out_wh)[0]
    if model == "sana":
        s = max(w / sw, h / sh)
        rw = max(w, round(sw * s))
        rh = max(h, round(sh * s))
        return fc.scale_crop_K(K, rw / sw, rh / sh, (rw - w) // 2, (rh - h) // 2)
    if model in ["lingbot", "worldwarp"]:
        if source_wh != (864, 480):
            raise ValueError("Legacy crop needs 864x480 source")
        left, cw = (16, 832) if model == "lingbot" else (72, 720)
        k = K.copy()
        k[0, 2] -= left
        return fc.scale_crop_K(k, w / cw, h / sh)
    return fc.scale_crop_K(K, w / sw, h / sh)


def reference_image(model, path, wh):
    im = np.asarray(Image.open(path).convert("RGB"))
    if model in ["lingbot", "worldwarp", "dreamx", "sana"]:
        from reference_crop import transform

        return transform(im, model, out_wh=wh)
    return fc.preprocess(model, im, wh)


def main(style="long"):
    p = argparse.ArgumentParser()
    for k in ["batch", "model", "scene", "out"]:
        p.add_argument("--" + k, required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument(
        "--horizon",
        type=int,
        default=80 if style == "gate" else 160 if style == "pair" else 480,
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--camera-only", action="store_true")
    p.add_argument("--check-inputs", action="store_true")
    a = p.parse_args()
    out = Path(a.out)
    batch = Path(a.batch)
    h = a.horizon
    if h < 1:
        raise ValueError("Horizon must be positive")
    if out.exists() and not a.check_inputs:
        raise FileExistsError(
            "Scores are immutable; select a new VIPE output path: " + str(out)
        )
    if style == "gate" and h != 80:
        raise ValueError("Scale gate uses exactly 80 generated frames")
    entries = [spec.split("=", 1) for spec in a.runs]
    if any(
        len(x) != 2 or not re.fullmatch(r"[A-Za-z0-9_-]+", x[0]) or not x[1]
        for x in entries
    ):
        raise ValueError("Use LABEL=PATH for each run")
    if len({x[0] for x in entries}) != len(entries):
        raise ValueError("Duplicate arm names")
    rows = [
        r
        for r in json.loads((batch / "index.json").read_text())["scenes"]
        if r["name"] == a.scene
    ]
    if len(rows) != 1:
        raise ValueError("Expected one matching scene in the batch index")
    row = rows[0]
    with np.load(batch / "trajectories" / f"{a.scene}.npz") as z:
        plan = np.linalg.inv(
            z["w2c_overrun" if "w2c_overrun" in z else "w2c"].astype(float)
        )[: h + 1]
        depth = z["anchor_depth"].astype(np.float32)
        intrinsics = z["intrinsics"].astype(float)
        K = intrinsics[0] if intrinsics.ndim == 3 else intrinsics
        if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
            raise ValueError("Expected finite, positive focal lengths in a 3x3 pixel K")
        if not np.allclose(intrinsics, K, rtol=0, atol=1e-5):
            raise ValueError(
                "The evaluator requires fixed intrinsics throughout the clip"
            )
    if len(plan) != h + 1:
        raise ValueError("Reference trajectory too short")
    source_wh = (depth.shape[1], depth.shape[0])
    fid = (
        sorted(
            set(
                np.linspace(1, h, 40).round().astype(int).tolist()
                + [x for x in [160, 320, h] if x <= h]
            )
        )
        if style == "long"
        else list(range(1, h + 1, 4))
    )
    shows = np.unique(np.linspace(0, h, 7 if style == "long" else 5).astype(int))
    results = {}
    saved = {}
    start = time.time()
    appearance = style != "gate" and not a.camera_only
    if appearance:
        if "frames_dir" not in row or "original_start" not in row:
            raise ValueError(
                "Appearance scores need frames_dir and original_start in the batch index. Use --camera-only for pose metrics."
            )
        references = batch / row["frames_dir"]
        for i in set(fid).union(shows):
            path = references / f"frame_{row['original_start']+int(i)+1:05d}.jpg"
            with Image.open(path) as image:
                if image.size != source_wh:
                    raise ValueError(
                        f"Recording frame {path} must match the anchor geometry {source_wh}"
                    )
    # Run each VIPE worker to completion before loading image-quality models.
    for arm, directory in entries:
        frames = load_frames(directory, h, a.scene)
        height, width = frames.shape[1:3]
        ko = output_K(a.model, K, source_wh, (width, height))
        rd = fc.remap_depth(depth, K, ko, (height, width))
        if a.check_inputs:
            continue
        trajectory = out.parent / (out.stem + "_trajectories") / (arm + ".npz")
        measured, meta = estimate(frames, ko, rd, trajectory)
        pe, re = errors(measured, plan)
        record = dict(
            camera_frame_ids=list(range(h + 1)),
            position_errors=pe.tolist(),
            rotation_errors=re.tolist(),
            measured_c2w=measured.tolist(),
            anchor_scale=meta["registration"]["scale"],
            vipe=meta,
            trajectory_file=str(trajectory),
            source=str(directory),
            finite=True,
            **path_metrics(measured, plan),
        )
        if style == "gate":
            sys.path.insert(0, str(A / "continuous_eval_20261029/scale_20261030"))
            from execution_metrics import execution_metrics

            ids = np.arange(0, 81, 4)
            record.update(
                execution_metrics(measured[ids], plan[ids]),
                measured_c2w=measured[ids].tolist(),
                commanded_c2w=plan[ids].tolist(),
                camera_frame_ids=ids.tolist(),
                position_errors=pe[ids].tolist(),
                rotation_errors=re[ids].tolist(),
                vipe_anchor_depth_scale=meta["registration"]["scale"],
            )
        elif appearance:
            saved[arm] = dict(
                gen=frames[fid].copy(), show=frames[shows].copy(), wh=(width, height)
            )
        if style != "gate":
            record.update(
                camera_mean=float(pe[1:].mean()),
                camera_endpoint=float(pe[-1]),
                rotation_mean=float(re[1:].mean()),
                rotation_endpoint=float(re[-1]),
            )
        results[arm] = record
        del frames
    if a.check_inputs:
        print(
            json.dumps(
                dict(
                    inputs_valid=True,
                    scene=a.scene,
                    horizon=h,
                    arms=[k for k, _ in entries],
                    appearance=appearance,
                )
            )
        )
        return
    if appearance:
        from evaluate_recorded import metrics_models
        import torch

        emb, lp = metrics_models()
        strips = {}
        for arm, _ in entries:
            item = saved[arm]
            refs = np.stack(
                [
                    reference_image(
                        a.model,
                        references / f"frame_{row['original_start']+i+1:05d}.jpg",
                        item["wh"],
                    )
                    for i in fid
                ]
            )
            ds = []
            ls = []
            with torch.no_grad():
                for i in range(0, len(fid), 4):
                    gen = item["gen"][i : i + 4]
                    ref = refs[i : i + 4]
                    ds.extend((emb(gen) * emb(ref)).sum(-1).cpu().tolist())
                    ls.extend(np.asarray(lp(gen, ref)).tolist())
            entry = results[arm]
            pe = np.array(entry["position_errors"])
            re = np.array(entry["rotation_errors"])
            horizons = {}
            for end in sorted(set([x for x in [160, 320, h] if x <= h])):
                mask = np.array(fid) <= end
                horizons[str(end)] = dict(
                    camera_mean=float(pe[1 : end + 1].mean()),
                    camera_endpoint=float(pe[end]),
                    rotation_mean=float(re[1 : end + 1].mean()),
                    rotation_endpoint=float(re[end]),
                    dino=float(np.array(ds)[mask].mean()),
                    lpips=float(np.array(ls)[mask].mean()),
                )
            entry.update(
                horizons=horizons,
                **horizons[str(h)],
                image_frame_ids=fid,
                dino_by_frame=ds,
                lpips_by_frame=ls,
            )
            strips.setdefault(
                "Recording",
                [
                    reference_image(
                        a.model,
                        references / f"frame_{row['original_start']+int(i)+1:05d}.jpg",
                        item["wh"],
                    )
                    for i in shows
                ],
            )
            strips[arm] = item["show"]
        sheet = Image.new("RGB", (280 * len(shows), 185 * len(strips)), "white")
        draw = ImageDraw.Draw(sheet)
        for ri, (label, frames) in enumerate(strips.items()):
            for ci, (i, frame) in enumerate(zip(shows, frames)):
                sheet.paste(
                    Image.fromarray(frame).resize((280, 158)), (280 * ci, 185 * ri + 25)
                )
                draw.text((280 * ci + 4, 185 * ri + 4), f"{label}: {i}", fill="black")
        sheet.save(out.with_suffix(".jpg"))
    report = dict(
        estimator="VIPE",
        evaluation_protocol="vipe_dense_anchor_depth_no_saturation_v2",
        model=a.model,
        scene=a.scene,
        seed=a.seed,
        horizon=h,
        appearance_scored=appearance,
        batch=str(batch),
        units="fixed reference anchor-depth units; not verified metric metres",
        camera_sampling="VIPE receives every consecutive frame; long/pair camera means use every generated frame",
        registration="anchor depth only; commanded trajectory never enters SLAM or gauge fitting",
        reference_provenance="stored batch poses; these may be VIPE-derived and are not asserted independent ground truth",
        arms=results,
        completed=time.time(),
        seconds=time.time() - start,
    )
    if style == "gate":
        report["frame_ids"] = list(range(0, 81, 4))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(out)
    print(
        json.dumps(
            {
                "out": str(out),
                "estimator": "VIPE",
                "arms": {
                    k: {
                        m: v.get(m)
                        for m in [
                            "camera_mean",
                            "camera_endpoint",
                            "rotation_mean",
                            "rotation_endpoint",
                            "dino",
                            "lpips",
                            "passed",
                        ]
                    }
                    for k, v in results.items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
