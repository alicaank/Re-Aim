# How Re-Aim corrects the next chunk

Re-Aim uses three camera poses at a chunk boundary. The target path specifies where the camera should be (`P`), a pose estimator measures where the generated camera arrived (`M`), and the world model may still have the last commanded camera (`L`) in its memory. The [controller](../src/camera_loop/controller.py) operates on camera-to-world poses in a common anchor frame and length unit.

## Re-Aim

1. Estimate the boundary pose from the images generated so far. Register the first pose to the anchor camera and use corresponding anchor depths to recover translation scale. Each adapter converts the result to its native camera convention. See [camera units](online_units.md).
2. Start the continuation from the measured pose. Across the next chunk, fade the position difference from the target to zero. Interpolate the orientation difference to identity using spherical linear interpolation. The resulting path `D` rejoins the target at the last requested frame.
3. If the model retains commanded camera labels, send `L @ inverse(M) @ D` for each future pose. This makes the requested displacement relative to the stored label equal the displacement from the measured camera to the corrected path. If the model has rebuilt its memory using measured poses, send `D` directly.

## Re-Aim + RGC

Rectified Gain Compensation estimates how strongly the generated camera responds along the commanded direction. At each observed boundary it projects measured translation onto commanded translation and measured rotation onto the commanded rotation axis, in the respective starting camera frames. For each channel it divides the cumulative projected motion by the cumulative commanded distance or angle. The gains start at one.

The controller divides the next chunk’s translation offsets and rotation vectors about the last commanded pose by their respective gains, each floored at one. For instance, camera motion twice as large as commanded gives a translation divisor of two. Motion perpendicular to the commanded translation contributes zero to its projected numerator. The [CPU example](../examples/reaim.py) demonstrates the same rule used by the six supported GPU adapters.
