"""Prepare the user's camera data and run the three matched comparisons."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .datasets import REQUIREMENTS, import_batch, validate_batch
from .experiments import execute_plan, plan_experiment
from .integrations import A, BUNDLE, RGC_ARM, materialize
from .registry import profile

PACKAGE = Path(__file__).resolve().parent
DA3 = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
DA3_REV = "b2359bdf726fb44ef62acca04d629dcf158053e7"
READY = A / "model_readiness_20261027"
ARMS = ["baseline", "correction", RGC_ARM]
# A marker is relative to the directory that will be mounted as the checkpoint.
MARKERS = {
    "sana": "sana_dit",
    "lyra": "lora/dmd_distillation.safetensors",
    "matrixgame": "first-person.safetensors",
    "worldplay": "generator_bf16.safetensors",
    "worldwarp": "worldwarp_latest.ckpt",
    "dreamx": "model.safetensors",
}
NAMES = {"dreamx", "dreamx-world-5b", "models--gd-ml--dreamx-world-5b"}


def checkpoints(root, selected):
    if not root.exists():
        raise ValueError(f"Checkpoint path does not exist: {root}")
    result = {}
    for model, marker in MARKERS.items():
        if selected and selected != model:
            continue
        if root.is_file():
            files = [root] if root.name == Path(marker).name else []
        else:
            files = list(root.rglob(Path(marker).name))
        found = set()
        for match in files:
            if model == "dreamx" and not (
                selected and (root.is_file() or match.parent == root)
            ):
                if not any(p.name.lower() in NAMES for p in match.parents):
                    continue
            candidate = match.parents[len(Path(marker).parts) - 1]
            if (candidate / marker).exists():
                found.add(candidate.resolve())
        # A repository may symlink its checkpoint into the upstream checkout layout.
        unique = {(folder / marker).resolve(): folder for folder in sorted(found)}
        if len(unique) > 1:
            raise ValueError(
                f"Multiple {model} checkpoints under {root}. Pass its specific model directory."
            )
        if unique:
            result[model] = next(iter(unique.values()))
    if not result:
        raise ValueError(
            f"No compatible checkpoint found at {root}. Expected markers: {MARKERS}"
        )
    return result


def link(source, destination):
    source = Path(source).resolve(strict=True)
    if destination.is_symlink() and destination.resolve() == source:
        return
    if (
        destination.exists()
        and destination.is_dir()
        and source.is_dir()
        and not destination.is_symlink()
    ):
        for item in source.iterdir():
            if not item.name.startswith("."):
                link(item, destination / item.name)
        return
    if destination.is_symlink() or destination.exists():
        raise ValueError(f"Checkpoint target is occupied: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def companion(search, repo, revision=None, only=None):
    from huggingface_hub import snapshot_download

    name = repo.split("/")[-1]
    for path in (search / name, search / repo, search):
        if path.name.lower() == name.lower() and path.is_dir():
            return path
    return Path(snapshot_download(repo, revision=revision, allow_patterns=only))


def mount_assets(w, model, primary, search, da3):
    repo = w / profile(model)["repo"]["path"]
    weights = w / READY / "weights"
    if model == "sana":
        link(primary, weights / "sana_streaming")
        link(
            companion(search, "Efficient-Large-Model/gemma-2-2b-it"), weights / "gemma2"
        )
    elif model == "lyra":
        link(primary, repo / "Lyra-2/checkpoints")
    elif model == "dreamx":
        link(primary, weights / "dreamx")
        link(
            companion(search, "Wan-AI/Wan2.2-TI2V-5B"),
            w / "FrameCrafter/models/Wan-AI/Wan2.2-TI2V-5B",
        )
    elif model == "matrixgame":
        dest = repo / "checkpoints"
        link(primary / "first-person.safetensors", dest / "first-person.safetensors")
        link(companion(search, "Wan-AI/Wan2.2-TI2V-5B"), dest / "Wan2.2-TI2V-5B")
        link(da3, dest / DA3.split("/")[-1])
    elif model == "worldwarp":
        dest = weights / "worldwarp"
        link(primary / "worldwarp_latest.ckpt", dest / "worldwarp_latest.ckpt")
        link(
            companion(search, "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
            dest / "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        )
        cut = companion(search, "liguang0115/cut3r", only=["cut3r_512_dpt_4_64.pth"])
        link(
            cut, w / "cache/huggingface/hub/models--liguang0115--cut3r/snapshots/local"
        )
    elif model == "worldplay":
        link(primary, w / "baselines/models/worldplay5b")
        link(
            companion(search, "Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
            w / "baselines/models/Wan2.2-TI2V-5B-Diffusers",
        )


def install_env(w, role, source):
    prefix = w.parent / "envs" / role
    python = prefix / "bin/python"
    marker = prefix / ".reaim-installed"
    if marker.exists():
        return python
    version = {"worldwarp": "3.12", "sana": "3.11", "observer": "3.11"}.get(
        role, "3.10"
    )
    if not (prefix / "conda-meta/history").is_file():
        packages = ["python=" + version, "pip", "cmake", "ninja", "packaging", "ffmpeg"]
        if role != "observer":
            cuda = {"worldwarp": "12.6", "worldplay": "12.4", "dreamx": "12.4"}.get(
                role, "12.8"
            )
            packages.append(f"cuda-toolkit={cuda}")
        if role == "lyra":
            packages += ["gcc_linux-64=13", "gxx_linux-64=13", "eigen", "zlib", "libgl"]
        subprocess.run(
            [
                "conda",
                "create",
                "-yp",
                prefix,
                "-c",
                "nvidia",
                "-c",
                "conda-forge",
                *packages,
            ],
            check=True,
        )
    da3_src = w / "lyra/Lyra-2/lyra_2/_src/inference/depth_anything_3"
    subprocess.run(
        [
            "conda",
            "run",
            "--no-capture-output",
            "-p",
            prefix,
            "bash",
            PACKAGE / "native_setup.sh",
            role,
            source,
            PACKAGE.parent,
            da3_src,
        ],
        check=True,
    )
    marker.touch()
    return python


def runtime(cache, model, ckpt, search):
    if not shutil.which("conda"):
        raise ValueError(
            "Install Conda or Miniforge and put conda on PATH to build the native GPU environments."
        )
    digest = hashlib.sha256()
    for source in (
        BUNDLE / "manifest.json",
        BUNDLE / "repos.json",
        BUNDLE / "patch_manifest.json",
        PACKAGE / "native_setup.sh",
        Path(__file__),
    ):
        digest.update(source.read_bytes())
    for path in (ckpt, search, Path(sys.executable)):
        digest.update(os.fsencode(path))
    root = cache / model / digest.hexdigest()[:16]
    root.mkdir(parents=True, exist_ok=True)
    w = root / "workspace"
    envs = root / "envs"
    observer = envs / "observer/bin/python"
    native = envs / model / "bin/python"
    # Serialize first-time materialization, installations and symlink creation.
    import fcntl

    with (root / "setup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (w / "camera_loop_workspace.json").is_file():
            if w.exists():
                # Preserve a failed checkout for diagnosis and retry in its final location.
                w.rename(
                    Path(tempfile.mkdtemp(dir=root, prefix="incomplete-")) / "workspace"
                )
            paths = {
                "@WORKSPACE@/baselines/venv_vmem": str(observer.parents[1]),
                "@CONDA_ROOT@/envs/lyra2": str(envs / "lyra"),
                "@CONDA_ROOT@/envs/cuda128": str(envs / "lyra"),
                "@WORKSPACE@/analysis/plan_base_restart_20261008/model_controls_20261024/bin/ffmpeg": str(
                    envs / "observer/bin/ffmpeg"
                ),
            }
            materialize(
                w,
                dict(
                    models=[model],
                    cache_root=str(w / "cache"),
                    path_map=paths,
                    python={model: str(native)},
                ),
                clone=True,
            )
        depth = companion(search, DA3, DA3_REV)
        link(
            depth,
            w
            / f"cache/huggingface/hub/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/{DA3_REV}",
        )
        mount_assets(w, model, ckpt, search, depth)
        install_env(
            w, "observer", w / "lyra/Lyra-2/lyra_2/_src/inference/depth_anything_3"
        )
        install_env(w, model, w / profile(model)["repo"]["path"])
    return w, native, observer, depth


def worldplay_calibration(primary):
    if os.environ.get("CAMERA_LOOP_UNIT_PROVIDER"):
        return None
    path = primary / "reaim-calibration.json"
    if not path.is_file():
        raise ValueError(
            f"Missing {path}: this HY-WorldPlay checkpoint needs its documented metres_per_native_unit and source, or a native camera-unit provider."
        )
    value = json.loads(path.read_text())
    factor = value.get("metres_per_native_unit")
    if (
        type(factor) not in (int, float)
        or not math.isfinite(factor)
        or factor <= 0
        or not str(value.get("source") or "").strip()
    ):
        raise ValueError(f"Invalid checkpoint calibration in {path}")
    return path


def input_kind(data):
    if (data / "index.json").is_file() and (data / "trajectories").is_dir():
        return "batch"
    if data.is_file() and data.name != "transforms.json":
        return "manifest"
    if (data / "scene.json").is_file() and not (data / "transforms.json").exists():
        return "manifest"
    return "DL3DV"


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="reaim",
        description="Run the world model, Re-Aim and Re-Aim + RGC on matched camera paths.",
    )
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Extracted DL3DV scene, dataset root, or registered batch",
    )
    p.add_argument(
        "--model", choices=list(MARKERS), help="Limit execution to one model"
    )
    p.add_argument(
        "--scene", help="Limit execution to one scene directory name or batch ID"
    )
    p.add_argument("--out", type=Path, default=Path("outputs/reaim"))
    p.add_argument(
        "--check",
        action="store_true",
        help="Validate inputs without installing software or generating videos",
    )
    p.add_argument("--generate-only", action="store_true", help="Skip video scoring")
    a = p.parse_args(argv)
    try:
        run(a)
    except (
        ValueError,
        KeyError,
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as exc:
        p.exit(1, f"reaim: {exc}\n")


def run(a):
    from .dl3dv import scenes, scene_frames, rectification, prepare_dl3dv
    from PIL import Image

    ckpt, data, out = a.checkpoints.resolve(), a.data.resolve(), a.out.resolve()
    models = checkpoints(ckpt, a.model)
    calibration = (
        worldplay_calibration(models["worldplay"]) if "worldplay" in models else None
    )
    kind = input_kind(data)
    count = max(max(REQUIREMENTS[m]) for m in models)
    files = []
    if kind == "DL3DV":
        files = [f for f in scenes(data) if not a.scene or f.parent.name == a.scene]
        if not files:
            raise ValueError(f"No matching scene {a.scene!r}")
        for f in files:
            spec, frames, paths = scene_frames(f, count)
            with Image.open(paths[0]) as im:
                rectification(spec, frames, im.size)
    print(
        f"Models: {', '.join(models)} | Input: {kind} | Arms: world model, Re-Aim, Re-Aim + RGC",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="reaim-check-") as temp:
        if not a.check:
            out.mkdir(parents=True, exist_ok=False)
        base = Path(temp) if a.check else out
        if kind == "manifest":
            batch = base / "batch"
            import_batch(
                data if data.is_file() else data / "scene.json", batch, resize=True
            )
        else:
            batch = data
        selected = [a.scene] if a.scene else None
        if kind != "DL3DV":
            for model in models:
                check = validate_batch(batch, model, selected)
                if not check["passed"]:
                    raise ValueError(json.dumps(check, indent=2))
        if a.check:
            print(
                "Input check passed. Model loading, calibration and CUDA execution require a generation run."
            )
            return
        cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "reaim"
        search = ckpt if ckpt.is_dir() else ckpt.parent
        for model, primary in models.items():
            w, native, observer, depth = runtime(cache, model, primary, search)
            if kind == "DL3DV" and batch == data:
                manifest = prepare_dl3dv(
                    files,
                    count,
                    out / "inputs",
                    observer,
                    PACKAGE / "anchor_depth.py",
                    depth,
                )
                batch = out / "batch"
                import_batch(manifest, batch)
            captions = None
            if model == "lyra":
                captions = out / "lyra-captions.pt"
                subprocess.run(
                    [
                        "bash",
                        w / A / "lyra2_plugins/run_t5_captions.sh",
                        batch,
                        captions,
                    ],
                    check=True,
                )
            plan = plan_experiment(
                w, model, native, batch, out / model, selected, ARMS, t5_cache=captions
            )
            prefix = native.parent.parent
            native_env = {
                "PATH": os.pathsep.join(
                    [str(native.parent), os.environ.get("PATH", "")]
                ),
                "CONDA_PREFIX": str(prefix),
                "CUDA_HOME": str(prefix),
                "PYTHONNOUSERSITE": "1",
            }
            if model == "worldplay" and calibration:
                native_env["REAIM_WORLDPLAY_CALIBRATION"] = str(calibration)
            execute_plan(plan, native_env=native_env)
            if not a.generate_only:
                from .review_score import evaluate

                evaluate(w, observer, plan, out / "scores" / model)
