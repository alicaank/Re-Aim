import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

RULE = Path(__file__).resolve().parents[1] / (
    "src/camera_loop/bundled/templates/analysis/plan_base_restart_20261008/continuous_eval_20261029/"
    "scale_20261030/correction_v4_gaincum_20261109/gain_rule.py")


def _rule():
    spec = importlib.util.spec_from_file_location("gain_rule", RULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pose(t, deg):
    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler("y", deg, degrees=True).as_matrix()
    m[:3, 3] = t
    return m


def test_translation_ratio_is_running_total():
    g = _rule().CumulativeGain(np.zeros(3), np.zeros(3))
    assert g.update(np.zeros(3), np.array([0, 0, 0.01]))["g"] == 1.0  # nothing commanded yet: keep 1
    rd1 = np.hypot(0.25, 0.01)  # measured step from (0, 0, 0.01) to (0.25, 0, 0)
    assert abs(g.update(np.array([0.1, 0, 0]), np.array([0.25, 0, 0]))["g"] - (0.01 + rd1) / 0.1) < 1e-9
    assert abs(g.update(np.array([0.2, 0, 0]), np.array([0.5, 0, 0]))["g"] - (0.01 + rd1 + 0.25) / 0.2) < 1e-9


def test_rotation_ratio_and_apply():
    g = _rule().CumulativeGainRT(np.eye(4), np.eye(4))
    for k in range(1, 4):  # commanded 0.1 forward and 2 deg per step; the model does 2.5x and 1.5x
        out = g.update(_pose([0, 0, 0.1 * k], 2 * k), _pose([0, 0, 0.25 * k], 3 * k))
    assert abs(out["g"] - 2.5) < 1e-9 and abs(out["g_rot"] - 1.5) < 1e-9
    ref = _pose([0, 0, 0.3], 6)
    new = g.apply(np.stack([_pose([0, 0, 0.3 + 0.1 * i], 6 + 2 * i) for i in (1, 2)]), ref)
    assert np.allclose(new[:, 2, 3] - 0.3, [0.04, 0.08])
    ang = [np.degrees(Rotation.from_matrix(m[:3, :3] @ ref[:3, :3].T).magnitude()) for m in new]
    assert np.allclose(ang, [4 / 3, 8 / 3])


def test_unit_gains_leave_commands_unchanged():
    g = _rule().CumulativeGainRT(np.eye(4), np.eye(4))
    cmds = np.stack([_pose([0.1, 0, 0.2], 5), _pose([0.2, 0, 0.3], 9)])
    assert np.allclose(g.apply(cmds, _pose([0, 0, 0.1], 1)), cmds)
