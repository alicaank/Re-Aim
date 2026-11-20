# Model adapters

The paper compares these six models on 52 DL3DV scenes. Use the same scene and seed for all three arms. The commands below use each adapter’s bundled pilot schedule.

| Model / upstream | Scale strategy in core | Native camera basis | State / limitations |
|---|---|---|---|
| [Matrix-Game 3.5](https://github.com/Riemann-Dynamics/Matrix-Game-3.5) (`matrixgame`) | `metric` | c2w / opencv | Metric metadata drives the adapter; native action encoding remains model-owned. Training corpus unresolved. |
| [Lyra-2](https://github.com/nv-tlabs/lyra) (`lyra`) | `anchor_depth` | w2c / opencv | Final native DA3/MoGe anchor depth; 96 history padding frames are not generated time. v3 quality gate awaits GPU validation. |
| [DreamX-World](https://github.com/AMAP-ML/DreamX-World) (`dreamx`) | `native_callback` | c2w / opencv | Camera indices are [0,1,5,...]; decoded feedback is delayed by three frames. Keep its adapter-specific alignment. |
| [WorldWarp](https://github.com/HyoKong/WorldWarp) (`worldwarp`) | `anchor_depth` | c2w / opencv | Initial TTT3R depth fit plus a two-chunk continuation gate; later chunks are unsupported by that gate. GPU validation pending. |
| [SANA-WM](https://github.com/NVlabs/Sana) (`sana`) | `metric` | c2w / opencv | Metric sidecar conversion, 24-frame chunks; preserve cached past camera conditioning. GPU validation pending. |
| [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) (`worldplay`) | `native_callback` | w2c / opencv | Native pose-unit contract unresolved. Explicit calibrated factor required; historical launcher lacks native/sham controls. |

## Install a baseline

Choose the model key from the table above. In a copy of `config.example.json`, set `models` to that key alone. Set the conda and cache roots to local directories and configure the native and observer interpreters as described in [setup](setup.md). Run this from the Re-Aim repository, using the CPU environment:

```bash
export WORKSPACE="$PWD/workspace"
camera-loop prepare --workspace "$WORKSPACE" --config config.local.json --clone
```

This creates the pinned sources and applies the bundled patches. Open the linked installation and checkpoint instructions for the **same revision** in the table below. Run the upstream installation commands inside its prepared source directory. Install CUDA/PyTorch and compiled extensions in that model's environment. Install the lightweight Re-Aim package in the native interpreter as well if the adapter imports shared helpers.

| CLI key | Directory under the workspace | Installation and checkpoint instructions | Re-Aim + RGC |
| --- | --- | --- | --- |
| `matrixgame` | `sota_repos/matrix_game/Matrix-Game-3.5` | [Pinned guide](https://github.com/Riemann-Dynamics/Matrix-Game-3.5/blob/fbf7def0693ae14f745ba35bf2a26215d4ef991d/README.md) | Shared rule wired in this release |
| `lyra` | `lyra` | [Pinned guide](https://github.com/nv-tlabs/lyra/blob/9fffc9adc37004091ecf26ef03abfb3abdf4d59a/Lyra-2/README.md) | Available |
| `dreamx` | `analysis/plan_base_restart_20261008/model_readiness_20261027/repos/dreamx` | [Pinned guide](https://github.com/AMAP-ML/DreamX-World/blob/a1f4c6e5e45600718e5236955f2e0702e53fc275/README.md) | Available |
| `worldwarp` | `analysis/plan_base_restart_20261008/model_readiness_20261027/repos/worldwarp` | [Pinned guide](https://github.com/HyoKong/WorldWarp/blob/0396b801c278546d01a6618a18e5a5c2a154c0a1/README.md) | Available |
| `sana` | `analysis/plan_base_restart_20261008/model_readiness_20261027/repos/sana` | [Pinned guide](https://github.com/NVlabs/Sana/blob/25adaaf02ad501c27be2c376b6b49b349cbcbe52/docs/sana_wm.md) | Available |
| `worldplay` | `sota_repos/worldplay` | [Pinned guide](https://github.com/Tencent-Hunyuan/HY-WorldPlay/blob/1588e1336e842b03b0a7860c654ebd7c46bb065e/README.md) | Shared rule wired in this release |

For SANA, run `bash ./environment_setup.sh sana` inside its prepared source directory. With the Hugging Face CLI installed and access to the listed checkpoints, download the weights to the directories the adapter reads:

```bash
weights="$WORKSPACE/analysis/plan_base_restart_20261008/model_readiness_20261027/weights"
hf download Efficient-Large-Model/SANA-WM_streaming --local-dir "$weights/sana_streaming"
hf download Efficient-Large-Model/gemma-2-2b-it --exclude gemma-2-2b-it.safetensors --local-dir "$weights/gemma2"
```

See [the asset inventory](../src/camera_loop/bundled/asset_download_plan.json) for other adapters. Install the DA3 observer dependencies from the prepared Lyra-2 depth submodule in the observer environment. To populate the exact checkpoint read by the worker, set `CACHE_ROOT` to your configured cache directory and run:

```bash
hf download depth-anything/DA3NESTED-GIANT-LARGE-1.1 --revision b2359bdf726fb44ef62acca04d629dcf158053e7 --cache-dir "$CACHE_ROOT/huggingface/hub"
```
The [Lyra-2 guide](rtx-5090.md) includes the observer setup and Lyra caption-cache step.

## Generate matched runs

For a first SANA pilot you can generate a calibrated input on CPU. The fixture includes reference frames with known intrinsics, depth and camera poses:

```bash
python examples/make_test_scene.py --out outputs/test-scene
export MODEL=sana BATCH="$PWD/outputs/test-scene/batch" SCENE=courtyard
```

This procedural scene is for checking installation and data flow. Use the benchmark scenes and recorded poses for paper comparisons.

Set `MODEL` to the CLI key, `MODEL_PYTHON` to the absolute path of the installed native interpreter, `BATCH` to a prepared batch, and `SCENE` to a name in its index. Run these commands in the Re-Aim CPU environment:

```bash
camera-loop doctor --workspace "$WORKSPACE" --config config.local.json --model "$MODEL"
camera-loop validate-batch --batch "$BATCH" --model "$MODEL" --scenes "$SCENE"
camera-loop run-batch --model "$MODEL" --python "$MODEL_PYTHON" --workspace "$WORKSPACE" \
  --batch "$BATCH" --scenes "$SCENE" --out "outputs/$MODEL" \
  --arms baseline correction correction_gainprojf --execute
```

Use `--arms baseline` for the unmodified command trajectory. For Lyra pass `--t5-cache /absolute/path/to/captions.pt` as well. Omit `--execute` to print the plan. Each run writes to a fresh directory under `outputs/MODEL/SCENE/ARM`. Native schedules and camera-unit requirements differ by model. HY-WorldPlay requires a native camera-unit provider before generation. The Matrix-Game and HY-WorldPlay adapters now apply the shared RGC rule at chunk boundaries. Their equivalence to the original 480-frame paper runs has not been verified. See [camera units](online_units.md) and [scoring](evaluation.md).
