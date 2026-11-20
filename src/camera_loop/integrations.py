"""Materialize isolated legacy/native adapters from source-only, hash-checked templates."""

import hashlib, json, os, re, subprocess
from pathlib import Path
from .registry import profiles, profile

BUNDLE = Path(__file__).with_name("bundled")
A = Path("analysis/plan_base_restart_20261008")
N = A / "new_models_loop_20261028"
C = A / "continuous_eval_20261029"
L = A / "long_domain_20261028"
F = A / "fair_common_20261025"
V = C / "scale_20261030/correction_v3_20261030"
GAIN_ADAPTERS = V.parent / "correction_v4_gaincum_20261109"
RGC_ARM = "correction_gainprojf"


def materialize(workspace, config, clone=False):
    w = Path(workspace).resolve()
    if w.exists() and any(w.iterdir()):
        raise ValueError(
            "Use an empty workspace; never materialize over experiments or installed models"
        )
    selected = config.get("models", list(profiles()))
    for m in selected:
        if "repo" not in profile(m):
            raise ValueError(f"{m} is a custom hook; supply its host model directly")
    values = {
        "@WORKSPACE@": str(w),
        "@HOME_ROOT@": str(Path(config.get("home_root", Path.home())).resolve()),
        "@CACHE_ROOT@": str(
            Path(config.get("cache_root", Path.home() / ".cache")).resolve()
        ),
        "@CONDA_ROOT@": str(
            Path(config.get("conda_root", Path.home() / "miniconda3")).resolve()
        ),
    }
    overrides = config.get("path_map", {})
    mappings = dict(values, **overrides)
    # The same model interpreter may also launch native subprocesses (including observer
    # subprocesses). A Python override must relocate those references too.
    for model, interpreter in config.get("python", {}).items():
        native = profile(model)["python"]
        key = native if native.startswith("@") else "@WORKSPACE@/" + native
        if not Path(interpreter).is_absolute():
            raise ValueError("Python overrides must be absolute paths")
        if key in overrides and overrides[key] != interpreter:
            raise ValueError(f"Conflicting interpreter/path_map override for {model}")
        mappings[key] = interpreter
    for key, value in mappings.items():
        if not isinstance(value, str) or any(
            c in value for c in ["\n", "\r", "'", '"', "\\"]
        ):
            raise ValueError(
                "Path mappings must be plain paths without quotes/backslashes/newlines"
            )
    pattern = "|".join(re.escape(k) for k in sorted(mappings, key=len, reverse=True))

    def render(s):
        return re.sub(pattern, lambda m: mappings[m.group()], s)

    entries = json.loads((BUNDLE / "manifest.json").read_text())
    # Verify all inputs before creating files or cloning anything.
    for item in entries:
        data = (BUNDLE / "templates" / item["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("Bundled source hash mismatch: " + item["path"])
    for path, expected in json.loads(
        (BUNDLE / "patch_manifest.json").read_text()
    ).items():
        if hashlib.sha256((BUNDLE / path).read_bytes()).hexdigest() != expected:
            raise ValueError("Bundled patch hash mismatch: " + path)
    w.mkdir(parents=True, exist_ok=True)
    pins = json.loads((BUNDLE / "repos.json").read_text())
    clone_models = set(selected) | ({"lyra", "vipe"} if selected else set())
    if clone:
        for model in sorted(clone_models):
            p = pins[model]
            target = w / p["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    p["url"],
                    str(target),
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(target), "checkout", "--detach", p["commit"]],
                check=True,
            )
            if model == "lyra":
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(target),
                        "submodule",
                        "update",
                        "--init",
                        "--recursive",
                        "--",
                        "Lyra-2/lyra_2/_src/inference/depth_anything_3",
                        "Lyra-2/lyra_2/_src/inference/vipe",
                    ],
                    check=True,
                )
            elif (target / ".gitmodules").exists():
                subprocess.run(
                    [
                        "git",
                        "-C",
                        str(target),
                        "submodule",
                        "update",
                        "--init",
                        "--recursive",
                    ],
                    check=True,
                )
            patch = render((BUNDLE / p["patch"]).read_text())
            if patch.strip():
                subprocess.run(
                    ["git", "-C", str(target), "apply", "--whitespace=nowarn", "-"],
                    input=patch,
                    text=True,
                    check=True,
                )
    for item in entries:
        src = BUNDLE / "templates" / item["path"]
        target = w / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        text = render(src.read_text())
        # Historical paths use both Path.home()/'.cache' and a single literal
        # containing the full Hugging Face subpath.
        text = re.sub(
            r"""Path\.home\(\)\s*/\s*(["'])\.cache(/[^'"]*)?\1""",
            lambda match: "Path("
            + repr(values["@CACHE_ROOT@"] + (match[2] or ""))
            + ")",
            text,
        )
        if target.name == "fair_common.py":
            text = text.replace(
                "BATCH = HERE / 'batch_ours10'",
                "BATCH = Path(os.environ.get('CAMERA_BATCH', str(HERE / 'batch_ours10')))",
            )
        target.write_text(text)
    # Refresh versioned snapshot hashes after relocation; source identity is retained in the bundle manifest.
    for adapter_dir in (V, GAIN_ADAPTERS):
        manifest_path = w / adapter_dir / "source_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for name, entry in manifest.items():
            entry["sha256"] = hashlib.sha256(
                (w / adapter_dir / "snapshots" / name).read_bytes()
            ).hexdigest()
        manifest_path.write_text(json.dumps(manifest, indent=2))
    receipt = dict(
        models=selected,
        workspace=str(w),
        paths=values,
        python=config.get("python", {}),
        bundle_manifest_sha256=hashlib.sha256(
            (BUNDLE / "manifest.json").read_bytes()
        ).hexdigest(),
        cloned=clone,
    )
    (w / "camera_loop_workspace.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def command(
    workspace,
    model,
    python,
    batch,
    scene,
    out,
    arm="correction",
    device="0",
    quality="enforce",
):
    if "repo" not in profile(model):
        raise ValueError(f"{model} is a custom hook; invoke it through AlignedLoop")
    w = Path(workspace).resolve()
    b = str(Path(batch).resolve())
    o = str(Path(out).resolve())
    rgc = arm == RGC_ARM
    version = GAIN_ADAPTERS if rgc else V
    if arm not in [RGC_ARM, "native", "baseline", "sham", "correction"]:
        raise ValueError("Unknown arm")
    if model == "worldplay" and arm not in {"baseline", "correction", RGC_ARM}:
        raise ValueError("WorldPlay supports baseline, Re-Aim and Re-Aim + RGC")
    if model == "lyra" and arm == "native":
        raise ValueError("Use the released Lyra entrypoint for native parity")
    if quality not in ["enforce", "report"]:
        raise ValueError("Invalid quality mode")
    common = ["--batch", b, "--scene", scene, "--out", o]
    corrected = {
        "correction": "replan_commanded",
        RGC_ARM: "replan_commanded_gainprojf",
    }.get(arm, arm)
    if model == "matrixgame":
        script = C / "mg35_stream_run.py"
        args = common + ["--arm", arm, "--blocks", "6", "--seed", "7"]
    elif model == "lyra":
        script = version / "lyra_v3.py"
        args = common + ["--arm", arm, "--frames", "481", "--observer-quality", quality]
    elif model == "worldwarp":
        script = version / "worldwarp_v3.py"
        args = common + [
            "--arm",
            arm,
            "--seed",
            "7",
            "--chunks",
            "2",
            "--scale_mode",
            "anchor",
            "--camera_window",
            "fixed",
            "--continuation-check",
            "enforce",
            "--observer-quality",
            quality,
        ]
    elif model == "sana":
        script = version / "sana_v3.py"
        args = common + [
            "--arm",
            arm,
            "--seed",
            "7",
            "--frames",
            "169",
            "--scale_mode",
            "anchor",
            "--observer-quality",
            quality,
        ]
    elif model == "worldplay":
        script = F / "wp_fair.py"
        args = [
            "--out",
            o,
            "--arms",
            str(w / A / "baseline_loop_20261028/arms.json"),
            "--arm",
            corrected,
            "--scenes",
            scene,
            "--seeds",
            "7",
        ]
    elif model == "dreamx":
        script = N / ("dreamx_run_gainproj.py" if rgc else "dreamx_run.py")
        args = common + ["--seed", "7", "--arm", arm, "--latents", "42"]
    else:
        raise ValueError(f"No native launcher for {model!r}")
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=device,
        LONG_BATCH=b,
        CAMERA_BATCH=b,
        GV_SCALE_CACHE=str(w / "camera_loop_calibration"),
        PYTHONUNBUFFERED="1",
    )
    return [str(python), str(w / script), *args], env


def doctor(workspace, models=None, python_overrides=None, conda_root=None):
    w = Path(workspace).resolve()
    found = []
    overrides = python_overrides or {}
    conda = Path(conda_root or Path.home() / "miniconda3")
    for name in models or profiles():
        p = profile(name)
        if "repo" not in p:
            found.append(dict(model=name, status=p["status"], note=p["notes"]))
            continue
        interpreter = overrides.get(
            name, p["python"].replace("@CONDA_ROOT@", str(conda))
        )
        interpreter = Path(interpreter)
        interpreter = interpreter if interpreter.is_absolute() else w / interpreter
        path = w / p["repo"]["path"]
        commit = None
        if (path / ".git").exists():
            r = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            )
            commit = r.stdout.strip() if r.returncode == 0 else None
        found.append(
            dict(
                model=name,
                status=p["status"],
                interpreter=str(interpreter),
                interpreter_exists=interpreter.is_file(),
                repo_exists=path.is_dir(),
                expected_commit=p["repo"]["commit"],
                actual_commit=commit,
                commit_matches=commit == p["repo"]["commit"],
                note=p["notes"],
            )
        )
    return found
