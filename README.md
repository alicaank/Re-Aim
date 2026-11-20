# Re-Aim: Correcting Trajectory Drift in Frozen Video World Models

Re-Aim estimates the camera pose after each generated video chunk and uses it to correct the next camera commands. Rectified Gain Compensation (RGC) measures how much the camera moved and turned along the commanded motion. When either gain exceeds one, RGC reduces the corresponding next command. The generator weights stay frozen.

## Try the controller on a CPU

From a Linux or macOS terminal with Git and Python 3.10–3.12, run:

```bash
git clone https://github.com/alicaank/closeloopcorrection.git
cd closeloopcorrection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
python examples/reaim.py
```

The example uses the packaged controller and the same gain rule as the native adapters. A simulated camera travels twice the commanded distance on a straight track. The controller observes the exact pose every ten frames. It needs no GPU, weights, dataset, or API key.

```text
Synthetic trajectory | mean position error (plan length = 1)
World model    0.5125
Re-Aim         0.1250
Re-Aim + RGC   0.0625
```

The numbers are averages over 40 simulated frames. The first chunk is shared by all three runs because the controller has yet to observe a completed chunk. For the complementary under-execution example, run `python examples/synthetic_loop.py`.

To save these poses and independently recompute the errors, follow the [CPU scoring example](docs/evaluation.md#check-a-trajectory-on-cpu).

## Verify the implementation

To test just the shared gain rule, use the environment above:

```bash
python -m pip install pytest
python -m pytest -q tests/test_gain_rule.py
```

The eight tests cover translation, rotation, sideways motion, overshoot, and the gain floor. For the full CPU suite on Linux, install FFmpeg on your PATH and run:

```bash
python -m pip install 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cpu
python -m pip install '.[test]'
python -m pytest -q -rs tests
```

The final summary gives the pass count and the reason for each skip. VMem parity requires its optional upstream source checkout. GitHub Actions also runs the suite and both examples on Python 3.10, 3.11, and 3.12.

## Generate and score videos

1. [Install a baseline](docs/models.md#install-a-baseline). Choose the model, prepare its pinned source, and install its native environment, checkpoints, and pose observer.
2. [Prepare a scene](docs/datasets.md). Use `python examples/make_test_scene.py --out outputs/test-scene` to generate a calibrated input and recording, or import your own first image, trajectory, intrinsics, and anchor depth. Include recording frames for appearance metrics.
3. [Generate matched runs](docs/models.md#generate-matched-runs). Use `baseline` for the world model, `correction` for Re-Aim, and `correction_gainprojf` for Re-Aim + RGC. Each comparison uses the same scene, seed, and horizon.
4. [Evaluate the outputs](docs/evaluation.md). The `camera-loop evaluate` command checks the inputs, estimates the generated camera trajectory with VIPE, and reports position error, rotation error, DINO similarity, and LPIPS. A camera-only mode works with a pose plan and generated frames.

RGC is exposed for SANA, Lyra-2, WorldWarp, Spatia, DreamX, and VMem. VMem's RGC arm writes commanded memory labels. Its separate measured-memory baseline uses `replan_measured`. Check the [adapter table](docs/models.md) for supported comparisons and calibration requirements.

The [controller](src/camera_loop/controller.py) implements the camera rejoin. The [CPU example](examples/reaim.py) shows how to use it with RGC. The [integration guide](docs/integration.md) describes camera conventions for adding a model. Upstream terms are in [NOTICE](NOTICE).
