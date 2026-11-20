"""Compare three camera controllers on a straight track with 2x motion gain."""

import numpy as np

from camera_loop import CameraAlignment, CorrectionSession, Measurement
from camera_loop.gain import ProjectedGainFloor


def run(mode):
    plan = np.repeat(np.eye(4)[None], 41, axis=0)
    plan[:, 0, 3] = np.linspace(0, 1, len(plan))
    actual = plan[0].copy()
    commanded = plan[0].copy()
    controller = CorrectionSession(CameraAlignment(1.0, evidence="simulation units"))
    gain = ProjectedGainFloor(commanded, actual)
    errors = []

    for start in range(0, 40, 10):
        ids = np.arange(start + 1, start + 11)
        commands = plan[ids]
        if start and mode != "World model":
            commands, _ = controller.correct(
                Measurement(actual.copy(), start, {"passed": True, "source": "exact simulated pose"}),
                planned_endpoint=plan[start],
                future_reference_c2w=commands,
                future_frame_ids=ids,
                available_last_frame=start,
                previous_native_command=commanded,
            )
            if mode == "Re-Aim + RGC":
                gain.update(commanded, actual)
                commands = gain.apply(commands, commanded)

        for frame, target in zip(ids, commands):
            actual[:3, 3] += 2 * (target[:3, 3] - commanded[:3, 3])
            commanded = target.copy()
            errors.append(np.linalg.norm(actual[:3, 3] - plan[frame, :3, 3]))

    return float(np.mean(errors))


if __name__ == "__main__":
    modes = ["World model", "Re-Aim", "Re-Aim + RGC"]
    errors = [run(mode) for mode in modes]
    print("Synthetic trajectory | mean position error (plan length = 1)")
    for mode, error in zip(modes, errors):
        print(f"{mode:14} {error:.4f}")
    assert errors[2] < errors[1] < errors[0]
