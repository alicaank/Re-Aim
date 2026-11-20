"""Single-image metric depth worker. Run with the isolated DA3 interpreter."""

import sys
import numpy as np
from PIL import Image


def main(image_path, intrinsics_path, checkpoint, output):
    import cv2
    from depth_anything_3.api import DepthAnything3

    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    K = np.load(intrinsics_path, allow_pickle=False).astype(np.float32)
    model = DepthAnything3.from_pretrained(checkpoint).to("cuda").eval()
    prediction = model.inference(image=[rgb], intrinsics=K[None], process_res=504)
    if not getattr(prediction, "is_metric", False):
        raise ValueError("Anchor depth requires the metric DA3 nested checkpoint")
    _, _, processed = model.input_processor(
        image=[rgb],
        intrinsics=K[None].copy(),
        process_res=504,
        process_res_method="upper_bound_resize",
    )
    target_K = processed[0].numpy()
    y, x = np.indices(rgb.shape[:2], dtype=np.float32)
    u = (x - K[0, 2]) * target_K[0, 0] / K[0, 0] + target_K[0, 2]
    v = (y - K[1, 2]) * target_K[1, 1] / K[1, 1] + target_K[1, 2]
    depth = cv2.remap(
        np.asarray(prediction.depth[0], np.float32),
        u,
        v,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    )
    np.save(output, depth)


if __name__ == "__main__":
    main(*sys.argv[1:])
