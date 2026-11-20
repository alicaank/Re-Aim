"""Exercise the installed scorer and the same example commands used in the README."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from camera_loop.integrations import materialize

ROOT = Path(__file__).resolve().parents[1]


def cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "camera_loop.cli", *map(str, args)],
        capture_output=True,
        text=True,
    )


def test_saved_pose_metrics(tmp_path):
    poses = tmp_path / "poses"
    subprocess.run(
        [sys.executable, str(ROOT / "examples/reaim.py"), "--save-poses", poses],
        check=True,
    )
    for arm, expected in [
        ("world-model", 0.5125),
        ("reaim", 0.125),
        ("reaim-rgc", 0.0625),
    ]:
        result = cli(
            "score-poses",
            "--target",
            poses / "target.npz",
            "--measured",
            poses / f"{arm}.npz",
        )
        assert result.returncode == 0, result.stderr
        metrics = json.loads(result.stdout)
        assert np.isclose(metrics["camera_mean"], expected)
        assert metrics["rotation_mean"] == 0
        assert metrics["generated_frames"] == 40


def test_evaluate_checks_generated_and_recorded_frames(tmp_path):
    scene = tmp_path / "scene"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples/make_test_scene.py"),
            "--out",
            scene,
            "--frames",
            "4",
        ],
        check=True,
    )
    workspace = tmp_path / "workspace"
    materialize(workspace, {"models": ["sana"]})
    frames = np.stack(
        [np.array(Image.open(p)) for p in sorted((scene / "recording").glob("*.jpg"))]
    )
    video = tmp_path / "frames.npz"
    np.savez(video, frames=frames)
    args = [
        "evaluate",
        "--workspace",
        workspace,
        "--batch",
        scene / "batch",
        "--scene",
        "courtyard",
        "--model",
        "sana",
        "--runs",
        f"reference={video}",
        "--out",
        tmp_path / "scores.json",
        "--check-inputs",
        "--horizon",
        "3",
    ]
    result = cli(*args)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["appearance"]
    assert not (tmp_path / "scores.json").exists()
    (scene / "recording/frame_00004.jpg").unlink()
    assert cli(*args).returncode != 0
    result = cli(*args, "--camera-only")
    assert result.returncode == 0 and not json.loads(result.stdout)["appearance"]
    assert cli(*args, "--camera-only", "--horizon", "4").returncode != 0
