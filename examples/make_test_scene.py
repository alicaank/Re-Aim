"""Build a calibrated two-plane scene for checking native launchers and scoring."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from camera_loop.datasets import import_batch


def make_scene(root, count):
    width, height, focal = 864, 480, 620
    root.mkdir(parents=True, exist_ok=False)
    recording = root / "recording"
    recording.mkdir()
    background = Image.new("RGB", (width + 160, height), "#e6d5b9")
    draw = ImageDraw.Draw(background)
    for y in range(0, height, 36):
        for x in range(-60 + (y // 36 % 2) * 40, background.width, 80):
            draw.rectangle((x, y, x + 76, y + 32), fill="#ad765a")
    for x in (80, 320, 560, 800):
        draw.rectangle((x, 64, x + 104, 224), fill="#263d45")
        draw.rectangle((x + 6, 70, x + 98, 152), fill="#96b3c1")
    foreground = Image.new("RGB", (144, 166), "#346854")
    slats = ImageDraw.Draw(foreground)
    for x in range(6, foreground.width, 16):
        slats.line((x, 5, x, 161), fill="#254537", width=3)

    # The camera moves parallel to both planes. Each plane shifts by f * dx / z.
    offsets = np.linspace(0.0, 0.18, count)
    w2c = np.repeat(np.eye(4)[None], count, axis=0)
    w2c[:, 0, 3] = -offsets
    intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    anchor_depth = np.ones((height, width), dtype=np.float32)
    anchor_depth[278:444, 520:664] = 0.5
    np.savez_compressed(
        root / "geometry.npz", w2c=w2c, intrinsics=intrinsics, anchor_depth=anchor_depth
    )
    for i, dx in enumerate(offsets):
        shift = round(focal * dx)
        image = background.crop((shift, 0, shift + width, height))
        image.paste(foreground, (520 - round(focal * dx / 0.5), 278))
        image.save(recording / f"frame_{i+1:05d}.jpg", quality=94)
    with Image.open(recording / "frame_00001.jpg") as anchor:
        anchor.save(root / "anchor.png")

    manifest = dict(
        dataset="procedural-two-planes",
        split="smoke-test",
        pose_convention="opencv_w2c_anchor_relative",
        depth_convention="z_depth_same_units_as_translation",
        scenes=[
            dict(
                name="courtyard",
                image="anchor.png",
                geometry="geometry.npz",
                caption="The camera moves right past a green crate in front of a brick wall.",
                prompts={
                    "astronex": "The camera moves right past a green crate in front of a brick wall.",
                    "worldplay": "The camera moves right past a green crate in front of a brick wall.",
                    "memorizon": "First-person perspective — character not visible. The camera moves right past a green crate.",
                },
                frames_dir="recording",
                original_start=0,
                metres_per_anchor_unit=4.0,
                metric_scale_evidence="Rendered wall at 4 metres and crate at 2 metres.",
            )
        ],
    )
    source = root / "scene.json"
    source.write_text(json.dumps(manifest, indent=2))
    import_batch(source, root / "batch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--frames", type=int, default=505, help="Total frames including the anchor"
    )
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("Choose at least two frames")
    make_scene(args.out, args.frames)
    print(f"Batch: {(args.out / 'batch').resolve()}\nScene: courtyard")
