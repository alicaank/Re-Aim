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


def test_projected_gain_counts_only_motion_along_the_command():
    g = _rule().ProjectedGain(np.eye(4), np.eye(4))
    # commanded 0.1 forward (+z) per step; the model goes 0.2 forward plus a sideways wobble of +-0.3 that alternates
    for k in range(1, 5):
        out = g.update(_pose([0, 0, 0.1 * k], 0), _pose([0.3 * (k % 2), 0, 0.2 * k], 0))
    assert abs(out["g"] - 2.0) < 1e-9          # the wobble does not count; CumulativeGain would report far more
    assert out["g_rot"] == 1.0                  # nothing turned yet: keep 1


def test_projected_gain_turning_ratio():
    g = _rule().ProjectedGain(np.eye(4), np.eye(4))
    for k in range(1, 4):                       # commanded 2 deg per step about y, the model turns 3 deg
        out = g.update(_pose([0, 0, 0.1 * k], 2 * k), _pose([0, 0, 0.1 * k], 3 * k))
    assert abs(out["g_rot"] - 1.5) < 1e-6


def test_floor_keeps_ratios_at_least_one():
    rule = _rule()
    g = rule.ProjectedGainFloor(np.eye(4), np.eye(4))
    for k in range(1, 4):                       # the model moves half as far and turns half as much as commanded
        out = g.update(_pose([0, 0, 0.1 * k], 2 * k), _pose([0, 0, 0.05 * k], 1 * k))
    assert out["g"] == 1.0 and out["g_rot"] == 1.0
    # steps are compared in each camera's own frame, and the two cameras turn differently: ~0.5, not exactly
    assert abs(out["g_raw"] - 0.5) < 1e-3 and abs(out["g_rot_raw"] - 0.5) < 1e-6
    cmds = np.stack([_pose([0.1, 0, 0.2], 5), _pose([0.2, 0, 0.3], 9)])
    assert np.allclose(g.apply(cmds, _pose([0, 0, 0.1], 1)), cmds)   # floor 1 = the original correction


def test_floor_shrinks_commands_when_the_model_overshoots():
    g = _rule().ProjectedGainFloor(np.eye(4), np.eye(4))
    for k in range(1, 4):
        out = g.update(_pose([0, 0, 0.1 * k], 0), _pose([0, 0, 0.25 * k], 0))
    assert abs(out["g"] - 2.5) < 1e-9
    new = g.apply(np.stack([_pose([0, 0, 0.4], 0)]), _pose([0, 0, 0.3], 0))
    assert np.allclose(new[0, 2, 3], 0.3 + 0.1 / 2.5)


def test_rules_table_names_every_arm():
    assert set(_rule().RULES) == {"correction_gaincum", "correction_gainrt", "correction_gainproj", "correction_gainprojf"}
