# correction_v4_gaincum_20261109: shared gain rule

Gain-aware adapters extend the v3 correction implementation.

The rule is one file, `gain_rule.py`, used by every adapter. It was written before any run that uses it.

- `correction_gaincum` (`CumulativeGain`): at every feedback point,
  g = total measured camera-centre movement since the start / total commanded movement since the start
  (sums of per-interval displacements in the model's native units). The next chunk's commanded translation offsets from
  the current command pose are divided by g. Rotation is unchanged.
- `correction_gainrt` (`CumulativeGainRT`): the same, plus
  g_rot = total measured rotation angle since the start / total commanded rotation angle since the start;
  rotation offsets (axis-angle of R_i R_ref^T, world frame) are divided by g_rot.

- `correction_gainproj` (`ProjectedGain`): like `correction_gainrt`, but each feedback interval counts only the measured
  motion along the commanded motion, so zero-mean measurement wobble (sideways or backwards) cancels instead of adding up:
  dc, dm = commanded and measured camera-centre steps, each in the camera's own frame at the interval start;
  g = sum(dm . dc/|dc|) / sum(|dc|). Turning works the same way with camera-frame rotation vectors (axis * angle).
  A ratio is kept when its total along the command is <= 0.
- `correction_gainprojf` (`ProjectedGainFloor`), **the rule we use**: `ProjectedGain` with both ratios floored at 1,
  g = max(1, g_projected). The rule can only shrink the next commands (the model moves or turns more than told), never
  enlarge them; where a model moves less than told it is exactly the original `correction`. Without the floor, a ratio
  below 1 enlarges the next commands, the model follows those even less, and the camera runs away from the plan (seen on
  DreamX, WorldWarp, HY-WorldPlay and MatrixGame). The raw ratios are logged as `g_raw` / `g_rot_raw`.

The projected rule uses the cumulative history. The rectified variant additionally floors each applied gain at one. A ratio keeps its previous value (starting at 1) only when it is
undefined: nothing commanded or measured yet, or a non-finite measurement. With g = g_rot = 1 the arm is exactly the
original `correction`.

`correction_gain` is the earlier per-model variant (window of 3 feedback intervals, minimum 0.05, clip [0.5, 4]; Lyra:
last chunk, minimum 0.02). Its settings were chosen after looking at test scenes. It is kept for reproducibility only.

Where the rule arms are wired in: Lyra-2 (`snapshots/lyra_plugin.py`), SANA (`snapshots/sana_run.py`), WorldWarp
(`snapshots/worldwarp_run.py`), Spatia (`snapshots/spatia_run.py`, arms `replan_commanded_gainproj` / `_gainprojf`),
DreamX (`new_models_loop_20261028/dreamx_run_gainproj.py`) and VMem (`long_domain_20261028/vmem_run_gainproj.py`, arms in
`vmem_arms_re10k.json`). WorldWarp and DreamX start every chunk at the measured pose, so their commanded interval runs from
the previous measured pose to the command for the newly measured frame.

Other switches (all off by default, so the arms above are unchanged):
- `GV_REJOIN=direct` (arm `correction_direct`): no fade; the new commands are the plan itself from the next frame on.
- `GV_LYRA_WRITE=predicted` (Lyra-2 arm `memmeas`): Lyra labels new memory frames with its own measured poses instead of
  the commanded ones.
- `GV_PLAN_KEY=w2c_overrun` (`spatia_v3.py`): Spatia reads the longer stored plan, e.g. 240-frame RealEstate10K runs.

Run the adapters directly, as with v3, e.g.
`python sana_v3.py --batch <batch> --scene <id> --out <dir> --arm correction_gainrt --seed 7 --frames 481 --scale_mode anchor`.
Use `camera-loop run --arm correction_gainprojf` for the rectified rule. The CLI keeps the historical launchers for baseline and ordinary correction.
