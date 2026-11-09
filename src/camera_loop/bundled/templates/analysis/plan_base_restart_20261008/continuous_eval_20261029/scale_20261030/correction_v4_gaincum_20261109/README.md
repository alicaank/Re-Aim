# correction_v4_gaincum_20261109: shared gain rule

Copy of `correction_v3_20261030` with gain-aware correction arms for Lyra-2 (`snapshots/lyra_plugin.py`) and SANA
(`snapshots/sana_run.py`). All other arms behave exactly as in `correction_v3_20261030`.

The rule is one file, `gain_rule.py`, used by every adapter. It was written before any run that uses it.

- `correction_gaincum` (`CumulativeGain`): at every feedback point,
  g = total measured camera-centre movement since the start / total commanded movement since the start
  (sums of per-interval displacements in the model's native units). The next chunk's commanded translation offsets from
  the current command pose are divided by g. Rotation is unchanged.
- `correction_gainrt` (`CumulativeGainRT`): the same, plus
  g_rot = total measured rotation angle since the start / total commanded rotation angle since the start;
  rotation offsets (axis-angle of R_i R_ref^T, world frame) are divided by g_rot.

There is no window, no minimum movement and no clipping. A ratio keeps its previous value (starting at 1) only when it is
undefined: nothing commanded or measured yet, or a non-finite measurement. With g = g_rot = 1 the arm is exactly the
original `correction`.

`correction_gain` is the earlier SANA-only variant (window of 3 feedback intervals, minimum 0.05, clip [0.5, 4]; Lyra:
last chunk, minimum 0.02). Its settings were chosen after looking at test scenes. It is kept for reproducibility only.

Run the adapters directly, as with v3, e.g.
`python sana_v3.py --batch <batch> --scene <id> --out <dir> --arm correction_gainrt --seed 7 --frames 481 --scale_mode anchor`.
`camera_loop.integrations` still points at `correction_v3_20261030`; these arms are not wired into the CLI yet.
