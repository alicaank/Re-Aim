# Runtime camera units

## Reference and native units

Keep the desired trajectory, measured poses, and controller residuals in one
reference coordinate system. Its initial depth must describe the input image in
the same units as the supplied desired trajectory. This initial-anchor contract
is allowed input; it is not a model-specific calibration or future observation.

For corresponding valid pixels, estimate

```
k = median(native_depth[p] / reference_depth[p])
t_native = k * t_reference
t_reference = t_native / k
```

Depths must share pixels and represent the same camera z-depth convention. The
fit requires at least 256 valid pixels, at least 20% coverage, positive finite
depths, and a p90/p10 ratio no larger than 4. These are geometry rejection gates,
not confidence intervals or demonstrated optimal thresholds. Receipts contain
coverage, dispersion, and median relative residual. The implementation does not
claim to detect every geometric inconsistency.

Do not substitute executed displacement / commanded displacement for `k`.
That ratio also contains the generator's execution response. Fitting it as units
can erase real under-execution from evaluation. Response learning remains a
separate controller quantity.

## Model interfaces

**WorldWarp.** `OnlineDepthCalibration` retains the initial anchor and the last
calibrated source frame. For the current 81-frame/one-frame-overlap interface,
chunk 0 uses the input image; chunk 1 reconstructs emitted frames 0–80; chunk 2
reconstructs 80–160, and so on. New native depth at the shared frame is fitted to
its retained reference depth. The new window's last depth divided by its own
`k` becomes the next reference. Shared-image mean absolute RGB disagreement must
be at most 8/255, allowing codec differences; this is not feature matching.

The desired path and observer anchor depth stay in reference units. Only a copy
of the camera tensor is multiplied by the current window's factor. Native
depth and reconstructed source poses already share that window's units.
Measured memory poses are made relative to the last source camera and then
multiplied by the same current factor exactly once. Rejection invalidates the
current factor; the old factor is never silently reused. Chained depth estimates
can accumulate error, so long-rollout GPU validation is still necessary.

**HY-WorldPlay.** Set `CAMERA_LOOP_UNIT_PROVIDER=module:function` only for an
audited runtime native-geometry implementation. The function is called afresh
per rollout with `model`, `anchor`, `intrinsics`, `frame_id=0`,
`available_frame=0`, and `anchor_sha256`. It receives no target path, recorded
reference video, or fitted scale. It must return:

```python
{
    "native_depth": depth_at_the_same_input_pixels,
    "source": "initial_input",
    "frame_id": 0,
    "anchor_sha256": received_anchor_sha256,
    "native_unit_evidence": "How runtime geometry matches this encoder's units",
    # optional: "mask": valid_correspondence_mask
}
```

Scalar-only answers and explicit `scale` fields are rejected. The interface
checks shapes, values, and image identity; it cannot independently prove that
an external provider's claimed native-unit semantics are true. A generic metric
depth model alone is insufficient unless the native metric conversion is known.

HY-WorldPlay requires an external provider that can identify native units from the anchor. This release includes the interface and validation checks. The native generation command stops if the provider is unavailable.
