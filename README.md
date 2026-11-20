# Re-Aim: Correcting Trajectory Drift in Frozen Video World Models

Re-Aim measures where the generated camera went and adjusts the next chunk of camera commands. Re-Aim + RGC also compensates when the camera moves or turns farther than commanded.

## Run on a scene

On Linux with an NVIDIA GPU, Git, Conda and Python 3.10–3.12:

```bash
git clone https://github.com/alicaank/Re-Aim.git
cd Re-Aim
python -m pip install .
reaim --checkpoints /path/to/model-weights --data /path/to/dl3dv-scene
```

Give the command an extracted DL3DV scene or a dataset directory. It detects the model, prepares its environment and camera inputs, generates all three comparisons and scores them under `outputs/reaim/`. The first run downloads upstream code and missing auxiliary weights. Add `--check` for a CPU input check, `--scene NAME` to select a scene, or `--out DIR` for another run.

The six models are **Lyra-2, SANA-WM, Matrix-Game 3.5, WorldWarp, DreamX-World and HY-WorldPlay**. HY-WorldPlay also needs its checkpoint’s camera-unit calibration. The bundled schedules range from 160 to 480 scored frames. [Checkpoint layouts and input requirements](docs/run.md)

To try the controller without weights or a GPU, run `python examples/reaim.py`.

[Method](docs/method.md) · [Paper results](docs/results.md) · [Evaluation](docs/evaluation.md)
