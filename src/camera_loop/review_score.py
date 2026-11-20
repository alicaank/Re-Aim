"""Score the three rollout paths recorded by the reviewer runner."""

import fcntl
import json
import os
from pathlib import Path
import subprocess

from .integrations import A


def evaluate(workspace, python, plan, out):
    vipe = workspace / "vipe"
    prefix = workspace.parent / "envs/vipe-build"
    env = dict(
        os.environ,
        CAMERA_VIPE_SOURCE=str(vipe),
        CAMERA_VIPE_PYTHON=str(vipe / ".venv/bin/python"),
        TORCH_HOME=str(workspace / "cache/torch"),
    )
    env.setdefault("HF_HUB_OFFLINE", "0")
    with (workspace.parent / "setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = prefix / ".reaim-ready"
        if not marker.is_file():
            if not (prefix / "conda-meta/history").is_file():
                subprocess.run(
                    [
                        "conda",
                        "env",
                        "create",
                        "-p",
                        prefix,
                        "-f",
                        vipe / "envs/cu128.yml",
                    ],
                    check=True,
                )
            subprocess.run(
                ["conda", "run", "-p", prefix, "uv", "sync"], cwd=vipe, check=True
            )
            marker.touch()

    out.mkdir(parents=True)
    rows = {
        row["name"]: row
        for row in json.loads((Path(plan["batch"]) / "index.json").read_text())[
            "scenes"
        ]
    }
    runs = {}
    for job in plan["jobs"]:
        scene, arm = job["scene"], job["arm"]
        path = Path(job["out"])
        if plan["model"] == "worldplay":
            names = {
                "baseline": "baseline",
                "correction": "replan_commanded",
                "correction_gainprojf": "replan_commanded_gainprojf",
            }
            path = path / names[arm] / f"{scene}_s{plan['seed']}.mp4"
        runs.setdefault(scene, []).append(f"{arm}={path}")
    if any("frames_dir" in rows[k] for k in runs):
        with (workspace.parent / "setup.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            marker = workspace / "cache/.appearance-ready"
            if not marker.exists():
                subprocess.run([python, "-m", "pip", "install", "lpips"], check=True)
                subprocess.run(
                    [
                        python,
                        "-c",
                        "import torch, lpips; torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vits14', trust_repo=True); lpips.LPIPS(net='alex')",
                    ],
                    env=env,
                    check=True,
                )
                marker.touch()
    horizon = 480 if plan["model"] in {"lyra", "matrixgame"} else 160
    for scene, sources in runs.items():
        cmd = [
            python,
            workspace / A / "vipe_eval_20261030/score.py",
            "--model",
            plan["model"],
            "--scene",
            scene,
            "--batch",
            plan["batch"],
            "--horizon",
            str(horizon),
            "--seed",
            str(plan["seed"]),
            "--out",
            out / f"{scene}.json",
            "--runs",
            *sources,
        ]
        if "frames_dir" not in rows[scene]:
            cmd.append("--camera-only")
        subprocess.run(cmd, env=env, check=True)
