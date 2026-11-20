# Run Re-Aim

`reaim --checkpoints WEIGHTS --data DATA` runs the baseline, Re-Aim and Re-Aim + RGC with seed 7. On its first run it needs network access, a CUDA-capable Linux machine compatible with the selected model, Git and Conda. Upstream builds and checkpoints need substantial disk space. Conda environments, patched sources and auxiliary assets are cached under `$XDG_CACHE_HOME/reaim` (default `~/.cache/reaim`). The input dataset and checkpoint directories are read-only.

## Checkpoints

Pass an unpacked model directory, or a parent containing multiple models. Use `--model KEY` to select one. Keep the upstream filenames and layouts. For a DreamX directory with a generic name, specify `--model dreamx`. The command identifies these files or folders:

| Key | Checkpoint marker |
| --- | --- |
| `lyra` | `checkpoints/lora/dmd_distillation.safetensors`, with the model, VAE and text encoder in the same checkpoint tree |
| `sana` | `sana_dit/` alongside the other streaming components |
| `matrixgame` | `first-person.safetensors` from Matrix-Game 3.5 Base |
| `worldwarp` | `worldwarp_latest.ckpt` |
| `dreamx` | `DreamX-World-5B/model.safetensors` |

The runner resolves shared Wan, Gemma, CUT3R and DA3 assets from the checkpoint directory or Hugging Face. Gated repositories require access in the account used by the Hugging Face client. The automatic runner skips HY-WorldPlay until its checkpoint calibration is available.

## Scene input

Use an extracted DL3DV scene containing Nerfstudio `transforms.json` and the corresponding numbered RGB frames (`images/`, `images_4/`, or another supplied resolution). A parent directory processes every scene unless you pass `--scene NAME`. Use the **images + poses** distribution. [Official DL3DV downloads](https://github.com/DL3DV-10K/Dataset#dataset-download)

The importer converts OpenGL camera poses to anchor-relative OpenCV poses, corrects lens distortion and maps RGB and intrinsics to the same grid. Offline preparation matches metric anchor depth from DA3 against triangulated correspondences in the recording. Calibration requires texture and parallax. Each rollout then receives its initial image, caption, target camera path and registered depth. If the scene lacks a caption, the command uses a generic camera-following prompt.

You can also point `--data` at an existing registered batch or a scene manifest. This is the route for exact paper calibration or a scene without enough parallax. The default adapters need 481 registered frames for Lyra, 505 for Matrix-Game, 169 for SANA, 161 for WorldWarp and 165 for DreamX. The scorer evaluates 480 frames for Lyra and Matrix-Game and 160 for the other three. [Registered batch format](datasets.md)

## Outputs

Videos or frame arrays are under `OUT/MODEL/SCENE/ARM/`. The plans and logs sit under `OUT/MODEL/` and the per-scene VIPE reports under `OUT/scores/MODEL/`. Appearance metrics use the corresponding recorded frames. Batches without recordings receive camera metrics only. Use `--generate-only` to skip scoring, or `--check` to validate input structure on CPU. Generation requires a fresh output directory. [Evaluation protocol](evaluation.md)
