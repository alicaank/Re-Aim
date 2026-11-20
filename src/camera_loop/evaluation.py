"""Score frame-matched camera-to-world trajectories in a shared anchor gauge."""

from pathlib import Path

import numpy as np

from .bundled.templates.analysis.plan_base_restart_20261008.vipe_eval_20261030.camera import (
    errors,
    path_metrics,
)
from .geometry import poses


def read_trajectory(path: Path):
    with np.load(path, allow_pickle=False) as data:
        trajectory = poses(data["c2w"])
        if trajectory.ndim != 3 or len(trajectory) < 2:
            raise ValueError("Expected at least two camera-to-world poses")
        if "frame_ids" in data and not np.array_equal(
            data["frame_ids"], np.arange(len(trajectory))
        ):
            raise ValueError(
                "Expected consecutive frame IDs starting at the conditioning frame"
            )
    return trajectory


def score_poses(target_path, measured_path):
    target = read_trajectory(target_path)
    measured = read_trajectory(measured_path)
    if target.shape != measured.shape:
        raise ValueError(
            "Target and measured trajectories must have matching frame counts"
        )
    position, rotation = errors(measured, target)
    return dict(
        generated_frames=len(target) - 1,
        camera_mean=float(position[1:].mean()),
        camera_endpoint=float(position[-1]),
        rotation_mean=float(rotation[1:].mean()),
        rotation_endpoint=float(rotation[-1]),
        **path_metrics(measured, target),
    )
