"""Exercise the two-path entry point without requiring model weights or CUDA."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from camera_loop.dl3dv import metric_scale, rectification, scene_frames
from camera_loop.reviewer import checkpoints, main, worldplay_calibration


def make_scene(directory, n):
    directory.mkdir()
    images = directory / "images_4"
    images.mkdir()
    poses = []
    for i in range(n):
        path = Path("images") / f"frame_{i + 1:05}.png"
        img = images / path.name
        if i == 0:
            Image.new("RGB", (80, 60)).save(img)
            first = img
        else:
            img.symlink_to(first)
        c2w = np.eye(4)
        c2w[0, 3] = i * 0.01
        poses.append(dict(file_path=str(path), transform_matrix=c2w.tolist()))
    spec = dict(w=320, h=240, fl_x=240, fl_y=240, cx=160, cy=120, frames=poses)
    path = directory / "transforms.json"
    path.write_text(json.dumps(spec))
    return path


def test_raw_directory_check_needs_only_two_paths(tmp_path):
    (tmp_path / "weights/sana_dit").mkdir(parents=True)
    path = make_scene(tmp_path / "room", 169)
    out = tmp_path / "out"
    main(
        [
            "--checkpoints",
            str(tmp_path / "weights"),
            "--data",
            str(path.parent),
            "--out",
            str(out),
            "--check",
        ]
    )
    assert not out.exists()
    spec, frames, images = scene_frames(path, 169)
    K, _ = rectification(spec, frames, (80, 60))
    np.testing.assert_allclose(K[:2, 2], [432, 240])
    assert images[0].parent.name == "images_4"
    (path.parent / "images_4/frame_00008.png").unlink()
    with pytest.raises(ValueError, match="Missing recording frame"):
        scene_frames(path, 169)


def test_metric_scale_on_known_parallel_motion(tmp_path):
    rng = np.random.default_rng(4)
    im = cv2.GaussianBlur(rng.integers(0, 256, (180, 260), np.uint8), (3, 3), 0)
    K = np.array([[180.0, 0, 130], [0, 180, 90], [0, 0, 1]])
    poses = np.repeat(np.eye(4)[None], 9, axis=0)
    paths = []
    for i in range(9):
        poses[i, 0, 3] = -i / 90
        path = tmp_path / f"{i}.png"
        # A fronto-parallel plane two scene units away shifts left one pixel per step.
        cv2.imwrite(str(path), np.roll(im, -i, axis=1))
        paths.append(path)
    assert metric_scale(paths, poses, K, np.full(im.shape, 6.0)) == pytest.approx(
        3.0, rel=0.02
    )
    with pytest.raises(ValueError, match="Insufficient triangulation"):
        metric_scale(
            paths, np.repeat(np.eye(4)[None], 9, axis=0), K, np.full(im.shape, 6.0)
        )


def test_checkpoint_discovery_and_calibration(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    (a / "first-person.safetensors").touch()
    (a / "model.safetensors").touch()
    assert checkpoints(tmp_path, None) == {"matrixgame": a}
    with pytest.raises(ValueError, match="Missing"):
        worldplay_calibration(a)
    p = a / "reaim-calibration.json"
    p.write_text(json.dumps(dict(metres_per_native_unit=-1, source="fixture")))
    with pytest.raises(ValueError, match="Invalid"):
        worldplay_calibration(a)


def test_run_routes_generation_and_scoring(tmp_path, monkeypatch):
    from camera_loop import reviewer
    from camera_loop import review_score
    from camera_loop.integrations import materialize

    scene = make_scene(tmp_path / "room", 169)
    weights = tmp_path / "SANA-WM_streaming"
    (weights / "sana_dit").mkdir(parents=True)
    w = tmp_path / "workspace"
    materialize(w, {"models": ["sana"]})
    monkeypatch.setattr(
        reviewer, "runtime", lambda *a: (w, Path("/usr/bin/python3"), "observer", "DA3")
    )
    # Replace only the native estimator and the model subprocesses.
    from camera_loop import dl3dv

    def depth_worker(argv, check):
        np.save(argv[-1], np.full((480, 864), 2.0, np.float32))

    monkeypatch.setattr(dl3dv, "subprocess", SimpleNamespace(run=depth_worker))
    monkeypatch.setattr(dl3dv, "metric_scale", lambda *a: 2.0)
    seen = []
    monkeypatch.setattr(reviewer, "execute_plan", lambda p, **kwargs: seen.append(p))
    monkeypatch.setattr(
        review_score, "evaluate", lambda w, py, p, out: seen.append(out)
    )
    out = tmp_path / "out"
    main(
        ["--checkpoints", str(weights), "--data", str(scene.parent), "--out", str(out)]
    )
    plan, scores = seen
    assert [j["arm"] for j in plan["jobs"]] == reviewer.ARMS
    assert scores == out / "scores/sana"
    row = json.loads((out / "batch/index.json").read_text())["scenes"][0]
    assert Path(row["frames_dir"]).joinpath("frame_00169.jpg").is_file()
    assert row["metres_per_anchor_unit"] == 2
