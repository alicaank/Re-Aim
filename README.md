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

The final summary gives the pass count and the reason for each skip. Two checks require optional upstream source checkouts. GitHub Actions also runs the suite and both examples on Python 3.10, 3.11, and 3.12.

## Run a video model

Native generation needs a supported GPU, the selected upstream model and observer environments, their checkpoints, and a calibrated scene. The [setup guide](docs/setup.md) covers source pins, environment paths and weights. The [dataset guide](docs/datasets.md) specifies the images, intrinsics, trajectories, and depth. For scoring generated videos, see the [VIPE evaluation guide](docs/evaluation.md).

For example, after preparing a SANA workspace and a batch, set `WORKSPACE`, `SANA_PYTHON`, `BATCH`, and `SCENE` to your local paths and scene ID. This command validates the inputs and prints the three native launch commands:

```bash
camera-loop run-batch --model sana --workspace "$WORKSPACE" --python "$SANA_PYTHON" \
  --batch "$BATCH" --scenes "$SCENE" --out outputs/sana-comparison \
  --arms baseline correction correction_gainprojf
```

Add `--execute` to generate the videos in a fresh output directory. This uses the bundled SANA pilot schedule. Full paper experiments additionally require the matching scenes, horizons, checkpoints and evaluation settings.

| Paper label | CLI arm | Behavior |
| --- | --- | --- |
| World model | `baseline` | Use the adapter's baseline commands. |
| Re-Aim | `correction` | Rejoin the planned path from the estimated camera pose. |
| Re-Aim + RGC | `correction_gainprojf` | Rejoin and divide translation and rotation offsets by their respective projected gains, each floored at one. |

The RGC launcher is available for SANA, Lyra-2, WorldWarp, Spatia, DreamX, and VMem. VMem's RGC arm uses commanded memory labels. Its measured-memory experiment is a separate comparison. See [model-specific conventions and limitations](docs/models.md).

## Code and evidence

- [CPU comparison](examples/reaim.py), [camera correction](src/camera_loop/controller.py), and [shared gain implementation](src/camera_loop/bundled/templates/analysis/plan_base_restart_20261008/continuous_eval_20261029/scale_20261030/correction_v4_gaincum_20261109/gain_rule.py)
- [Method reference](docs/paper_method_reference.md), [integration API](docs/integration.md), and [validation protocol](docs/validation.md)
- [Historical model results](docs/results.md), [failures](docs/failures.md), and [ablations](docs/ablations.md)
- [Upstream notices](NOTICE)
