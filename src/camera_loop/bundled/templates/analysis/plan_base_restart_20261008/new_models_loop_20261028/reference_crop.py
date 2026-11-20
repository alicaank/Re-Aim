"""Reference RGB and depth transforms used by the paired evaluator."""

import numpy as np
from PIL import Image


def transform(x, model, depth=False, out_wh=None):
    if model == "lingbot":
        im = Image.fromarray(
            np.asarray(x, dtype=np.float32)[:, 16:848]
            if depth
            else np.asarray(x)[:, 16:848]
        )
        return np.asarray(
            im.resize(
                out_wh or (832, 464),
                Image.Resampling.BILINEAR if depth else Image.Resampling.BICUBIC,
            )
        )
    if model == "worldwarp":
        cropped = Image.fromarray(np.asarray(x)[:, 72:792])
        return np.asarray(
            cropped.resize(out_wh or cropped.size, Image.Resampling.BILINEAR)
        )
    im = Image.fromarray(np.asarray(x, dtype=np.float32) if depth else np.asarray(x))
    method = Image.Resampling.BILINEAR
    if model == "sana":
        sw, sh = im.size
        width, height = out_wh or (1280, 704)
        scale = max(height / sh, width / sw)
        rw = max(width, round(sw * scale))
        rh = max(height, round(sh * scale))
        left = (rw - width) // 2
        top = (rh - height) // 2
        return np.asarray(
            im.resize((rw, rh), method if depth else Image.Resampling.LANCZOS)
        )[top : top + height, left : left + width]
    if model in ["dreamx", "matrixgame"]:
        return np.asarray(im.resize(out_wh or (1280, 704), method))
    raise ValueError(model)
