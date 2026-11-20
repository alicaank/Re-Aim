# Earlier baseline and Re-Aim comparisons

These runs compare the frozen world model with the first Re-Aim implementation. They predate RGC and use the cohorts and calibration protocols listed below. The VMem run also updates its memory with measured camera poses. The table reports the relative change in mean position error across matched scenes. A negative value means lower error.

| Model / cohort | Seed | Valid position pairs | Mean position change |
| --- | ---: | ---: | ---: |
| DreamX / RE10K160 | 7 | 10 | -32.5% |
| LingBot / RE10K160 | 7 | 10 | -46.1% |
| LingBot / RE10K160 | 107 | 6 | -29.6% |
| MatrixGame / RE10K160 | 7 | 10 | -7.4% |
| MatrixGame / DL3DV480 | 7 | 7 | -43.0% |
| Astronex / MinWM76 | 7 | 5 | -10.7% |
| VMem / RE10K40, commands + measured memory | 7 | 10 | -25.1% |
| Lyra / VIPE v3, 480 | 7 | 5 | -54.5% |
| SANA / VIPE v3, 160 | 7 | 9 | -52.3% |
| Spatia / VIPE v3, 480 | 7 | 7 | +6.1% |

Of 80 historical pairs, 79 had valid position estimates for both arms. One SANA comparison lacked a position estimate. In 21 pairs, Re-Aim increased the mean position error. Eight more showed a tradeoff between mean and endpoint error. The Astronex run used reference motion for calibration.

Tracking scores can also miss a change in scene layout. In DreamX and SANA scene `e43058cb` (seed 7), both arms missed the planned kitchen traversal. In MatrixGame scene `0238de14`, both videos showed a seam that was partly attributable to decoding. When inspecting a new comparison, review the generated video alongside the per-frame trajectory and recording. See the [evaluation guide](evaluation.md).
