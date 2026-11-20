# Evaluate camera trajectories and generated videos

## Check a trajectory on CPU

From the repository root, activate the quick-start environment and run:

```bash
python examples/reaim.py --save-poses outputs/cpu-poses
camera-loop score-poses --target outputs/cpu-poses/target.npz --measured outputs/cpu-poses/reaim-rgc.npz
```

The synthetic RGC run gives a mean position error of `0.0625` and zero rotation error. Substitute `world-model.npz` or `reaim.npz` to score the other two runs. Both inputs contain a `c2w` array of shape `[frames,4,4]` with corresponding frame IDs. They must already share the anchor frame, camera convention and length unit. For a saved VIPE trajectory use its registered `c2w` array. The scorer compares corresponding poses directly.

## Install the video evaluator (Linux with CUDA)

Use the workspace created by `camera-loop prepare --clone`. If you already have generated videos and only need the evaluation tools, prepare a fresh workspace with one selected model using the [setup instructions](models.md#install-a-baseline). The evaluator uses two interpreters. VIPE reconstructs poses in a GPU subprocess. A separate scoring interpreter reads frames and computes the metrics.

1. Build VIPE inside its pinned checkout. The upstream CUDA 12.8 recipe uses the following commands. Run these in a subshell to keep your control environment active afterwards.

   ```bash
   ( cd "$WORKSPACE/vipe" && conda env create -f envs/cu128.yml && conda run -n cu128 uv sync )
   export CAMERA_VIPE_SOURCE="$WORKSPACE/vipe"
   export CAMERA_VIPE_PYTHON="$WORKSPACE/vipe/.venv/bin/python"
   ```

   Use the [pinned VIPE installation guide](https://github.com/nv-tlabs/vipe/blob/95a8816947602ddc26fcb7a80bea4f9313059578/docs/installation.md) for compiler and system prerequisites. Populate VIPE's default SLAM and `unidepth-l` weights in the worker's Hugging Face cache before scoring. The evaluation worker runs offline. To check that its environment imports and parses the actual pipeline configuration, run:

   ```bash
   "$CAMERA_VIPE_PYTHON" "$WORKSPACE/analysis/plan_base_restart_20261008/vipe_eval_20261030/worker.py" --dry-run
   ```

2. Create the scoring environment. For camera-only scores, NumPy, SciPy, Pillow and OpenCV suffice for the parent process. The VIPE worker still requires CUDA.

   ```bash
   python -m venv .venv-score
   export SCORE_PYTHON="$PWD/.venv-score/bin/python"
   "$SCORE_PYTHON" -m pip install numpy scipy pillow opencv-python-headless
   ```

3. For appearance scores, install matching CUDA builds of Torch and torchvision plus LPIPS in the scoring environment. The following example uses CUDA 12.8. Set `CACHE_ROOT` to the same absolute `cache_root` used during workspace preparation. Run the last two commands with network access to cache DINOv2 ViT-S/14 source and weights and the LPIPS AlexNet weights before the offline evaluation.

   ```bash
   "$SCORE_PYTHON" -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
   "$SCORE_PYTHON" -m pip install lpips
   export TORCH_HOME="${CACHE_ROOT:?set CACHE_ROOT to the configured cache root}/torch"
   "$SCORE_PYTHON" -c "import torch; torch.hub.load('facebookresearch/dinov2:main', 'dinov2_vits14', trust_repo=True)"
   "$SCORE_PYTHON" -c "import lpips; lpips.LPIPS(net='alex')"
   ```

   Torch Hub stores the DINOv2 source under `torch/hub/facebookresearch_dinov2_main` and the checkpoint under `torch/hub/checkpoints`. Record the environment versions and hashes of the weights and cached source when sharing scores.

## Score the generated frames

The example below follows the SANA run from the [model guide](models.md#generate-matched-runs). Set `WORKSPACE`, `BATCH`, `SCENE`, and `SCORE_PYTHON` as above. SANA's pilot requests 169 frames. A horizon of 160 scores the first 160 generated frames after the conditioning frame.

```bash
runs="outputs/sana/$SCENE"
args=(--workspace "$WORKSPACE" --python "$SCORE_PYTHON" --model sana
      --batch "$BATCH" --scene "$SCENE" --seed 7 --horizon 160
      --out "scores/sana/$SCENE.json"
      --runs "baseline=$runs/baseline" "correction=$runs/correction"
             "correction_gainprojf=$runs/correction_gainprojf")
camera-loop evaluate "${args[@]}" --check-inputs
camera-loop evaluate "${args[@]}"
```

The first command checks that the scorer can read each rollout and the sampled reference frames on CPU. The second runs VIPE and the image metrics. To score camera motion alone, add `--camera-only` to both commands. To score two arms, supply two `LABEL=PATH` arguments instead of three. Use a fresh output stem for each attempt.

For another model, change the model key, horizon, and paths. Point each label directly at a `frames.npz` file containing `uint8 [N,H,W,3]` RGB, a video file, or a directory containing `frames.npz`, `SCENE_frames/frame_00000.png` onward, or `review_full.mp4`. Some adapters place videos in a nested arm directory. Each source must start at the conditioning frame and have at least `horizon + 1` consecutive frames from the same scene, seed and camera plan. Appearance scoring requires the corresponding recording frames described in the [dataset guide](datasets.md).

## Read the output

The JSON report records the estimator, protocol, seed, frame IDs, per-frame errors and per-arm summaries. Each arm also has a saved registered trajectory, raw VIPE trajectory, worker metadata and log in the adjacent `*_trajectories/` directory. Appearance runs produce a contact sheet.

| Quantity | Definition |
| --- | --- |
| Position error | Euclidean distance between estimated and planned camera centers in reference anchor-depth units. |
| Rotation error | Angle in degrees between estimated and planned orientations. |
| Mean and endpoint | The mean covers all generated frames and excludes the anchor. The endpoint uses the final scored frame. |
| Path-length ratio | Measured distance traveled divided by planned distance traveled. Undefined for a stationary translation plan. |
| DINO / LPIPS | Cosine similarity of normalized DINOv2 embeddings (higher is better) and LPIPS AlexNet distance (lower is better) against the recorded view. The long scorer samples up to 40 evenly spaced frame IDs plus available reporting horizons at 160, 320 and the final frame. |

The video protocol is `vipe_dense_anchor_depth_no_saturation_v2`. VIPE sees generated RGB and intrinsics. The scorer aligns the first pose and estimates one translation scale from registered anchor depth, excluding saturated depth estimates. It requires at least 256 valid pixels, 20% shared coverage, and a 90th-to-10th percentile depth-ratio spread of at most four. Inspect estimation failures and record the denominator when averaging across scenes. WorldScore uses a separate evaluation pipeline. Its paper-specific runner and configuration are absent from this release.
