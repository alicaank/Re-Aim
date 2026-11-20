# Camera tracking on DL3DV

Mean position error over the 52 DL3DV scenes (480 frames) reported on the [project page](https://alicaank.github.io/camera-feedback/#results-h). Lower values indicate closer tracking.

| World model | Baseline | Re-Aim | Re-Aim + RGC |
| --- | ---: | ---: | ---: |
| Lyra-2 | 0.083 | 0.029 | 0.017 |
| SANA-WM | 1.032 | 0.142 | 0.056 |
| Matrix-Game 3.5 | 0.157 | 0.039 | 0.046 |
| WorldWarp | 0.263 | 0.202 | 0.190 |
| DreamX-World | 0.229 | 0.069 | 0.078 |
| HY-WorldPlay | 1.066 | 0.046 | 0.060 |

These are the published measurements. The [model guide](models.md) lists the bundled pilot schedules and [evaluation](evaluation.md) describes how to score new runs.
