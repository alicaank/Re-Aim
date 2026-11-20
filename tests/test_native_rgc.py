"""Check the two native command paths with a camera that moves twice as far as asked."""
import ast
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from camera_loop.gain import ProjectedGainFloor
from camera_loop.integrations import A, BUNDLE


SOURCES = BUNDLE / "templates" / A


def native_function(path, name, scope):
    # Execute the shipped callback on CPU without importing the CUDA model.
    node = next(n for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    exec(compile(ast.Module([node], []), str(path), "exec"), scope)
    return scope[name]


def track(n):
    plan = np.repeat(np.eye(4)[None], n, axis=0)
    plan[:, 0, 3] = np.arange(n)
    return plan


def test_matrix_scales_future_commands_after_rejoin(tmp_path):
    plan = track(169)
    boundary = 84
    measured = plan[boundary].copy()
    measured[0, 3] *= 2
    worker = Mock()
    worker.measure.return_value = measured, {}
    common = SOURCES / "new_models_loop_20261028/common.py"
    from scipy.spatial.transform import Rotation, Slerp
    rejoin = native_function(common, "rejoin", dict(np=np, Rotation=Rotation, Slerp=Slerp))
    scope = dict(
        np=np, torch=SimpleNamespace(tensor=torch.tensor, random=SimpleNamespace(fork_rng=lambda **_: nullcontext())),
        worker=None, PoseWorker=lambda _: worker, arm="correction_gainprojf", plan=plan,
        prefix=np.zeros((85, 2, 2, 3), np.uint8), K=np.eye(3), depth=np.ones((2, 2)),
        out=tmp_path, events=[], all_commands=plan.copy(), native_w2c=np.linalg.inv,
        rejoin=rejoin, gain=ProjectedGainFloor(plan[0], plan[0]), hashlib=hashlib, json=json,
    )
    camera = dict(noisy_latent_indices_prope_extrinsic=torch.tensor(np.linalg.inv(plan[85:])),
                  clean_latent_indices_prope_extrinsic=torch.tensor(np.linalg.inv(plan[81:85])))
    observe = native_function(SOURCES / "continuous_eval_20261029/mg35_stream_launch.py", "observe", scope)
    observe(1, Mock(), camera)
    base = plan[boundary] @ np.linalg.inv(measured) @ rejoin(measured, plan[boundary], plan[85:])
    np.testing.assert_allclose(scope["all_commands"][85:, 0, 3], 84 + (base[:, 0, 3] - 84) / 2)
    np.testing.assert_array_equal(scope["all_commands"][:85], plan[:85])
    np.testing.assert_allclose(camera["noisy_latent_indices_prope_extrinsic"], np.linalg.inv(scope["all_commands"][85:]))
    assert scope["events"][0]["measurement"]["rgc"]["g"] == 2


def test_worldplay_scales_next_chunk_before_native_camera_encoding():
    plan = track(8)
    pipe = Mock()
    rgb = np.zeros((1, 4, 2, 2, 3), np.float32)
    pipe.decode_next_latent.return_value = rgb
    observed = plan.copy()
    observed[:, 0, 3] *= 2
    common = SOURCES / "fair_common_20261025/fair_common.py"
    interp = native_function(common, "se3_interp", dict(np=np))
    fc = SimpleNamespace(se3_interp=interp, da3_window=lambda *args, **kw: dict(poses=observed[1:], scale=1))
    scope = dict(np=np, torch=torch, C=4, H=2, W=2, fc=fc)
    run = native_function(common.with_name("wp_fair.py"), "run", scope)
    roller = SimpleNamespace(wp=SimpleNamespace(pipe=pipe, prompt=lambda _: (None,None)), seed=7, chunks=2,
                             plan=plan, cmd=plan.copy(), mem=plan.copy(), measured=[plan[0]]+[None]*7,
                             gain=ProjectedGainFloor(plan[0],plan[0]), arm={"replan":True}, events=[],
                             text="fixture", image=None, anchor=rgb, anchor_depth=np.ones((2,2)), K=np.eye(3),
                             da3=Mock(), unit=1, unit_calibration={})
    rows = []
    def encode(lo,hi):
        rows.append(roller.mem[lo:hi].copy())
        return None, None, None
    roller.wp_rows = encode
    run(roller)
    np.testing.assert_array_equal(rows[0], plan[:4])
    # At the second boundary the original command is x=3 and the measurement is x=6.
    # Re-Aim rejoins x=7 by the last frame. A gain of two halves that next offset.
    assert rows[1][-1, 0, 3] == 5
    assert next(e for e in roller.events if e["kind"] == "rgc")["g"] == 2
