import compileall, hashlib, json, os, tempfile, unittest, subprocess, sys
from pathlib import Path
from camera_loop.integrations import materialize, command, BUNDLE, V
from camera_loop.registry import profiles


class Integrations(unittest.TestCase):
    def test_template_inventory_matches_manifest(self):
        expected = {item['path'] for item in json.loads((BUNDLE/'manifest.json').read_text())}
        actual = {str(p.relative_to(BUNDLE/'templates')) for p in (BUNDLE/'templates').rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts}
        self.assertEqual(actual, expected, 'Stale or untracked adapter files in source/wheel')

    def test_patch_integrity_and_syntax(self):
        for path, expected in json.loads(
            (BUNDLE / "patch_manifest.json").read_text()
        ).items():
            data = (BUNDLE / path).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), expected)
            if data.strip():
                result = subprocess.run(
                    ["git", "apply", "--numstat", "-"], input=data, capture_output=True
                )
                self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_relocation_compile_and_v3_manifests(self):
        with tempfile.TemporaryDirectory() as td:
            w = Path(td) / "work"
            materialize(
                w,
                {
                    "home_root": "/home/teammate",
                    "conda_root": "/opt/conda",
                    "cache_root": "/data/cache",
                },
            )
            self.assertTrue(compileall.compile_dir(str(w), quiet=1))
            for p in w.rglob("*.py"):
                s = p.read_text()
                for token in ["@WORKSPACE@", "/home/hj", "@CONDA_ROOT@"]:
                    self.assertFalse(token in s, f"{token} remains in {p}")
            m = json.loads((w / V / "source_manifest.json").read_text())
            for name, record in m.items():
                self.assertEqual(
                    hashlib.sha256(
                        (w / V / "snapshots" / name).read_bytes()
                    ).hexdigest(),
                    record["sha256"],
                )
            with self.assertRaises(ValueError):
                materialize(w, {})

    def test_six_paper_launchers_materialize(self):
        self.assertEqual(set(profiles()), set(['lyra', 'sana', 'matrixgame', 'worldwarp', 'dreamx', 'worldplay']))
        with tempfile.TemporaryDirectory() as td:
            w = Path(td) / "work"
            materialize(w, {})
            for model in profiles():
                argv, env = command(
                    w, model, "/opt/python", "/data/batch", "scene", "/data/out"
                )
                self.assertTrue(Path(argv[1]).is_file(), model)
                self.assertEqual(env["LONG_BATCH"], "/data/batch")
                if model in ["lyra", "sana", "worldwarp"]:
                    self.assertIn("--observer-quality", argv)

    def test_rectified_gain_launchers_in_relocated_workspace(self):
        expected = {
            "sana": ("sana_v3.py", "correction_gainprojf"),
            "lyra": ("lyra_v3.py", "correction_gainprojf"),
            "worldwarp": ("worldwarp_v3.py", "correction_gainprojf"),
            "matrixgame": ("mg35_stream_run.py", "correction_gainprojf"),
            "dreamx": ("dreamx_run_gainproj.py", "correction_gainprojf"),
            "worldplay": ("wp_fair.py", "replan_commanded_gainprojf"),
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            materialize(root, {})
            for model, (script, arm) in expected.items():
                with self.subTest(model=model):
                    argv, _ = command(root, model, sys.executable, root / "batch",
                                      "scene", root / "out", "correction_gainprojf")
                    launcher = Path(argv[1])
                    self.assertTrue(launcher.is_file())
                    self.assertEqual(launcher.name, script)
                    self.assertIn(arm, argv)
                    if model in {"sana", "lyra", "worldwarp"}:
                        self.assertTrue((launcher.parent / "gain_rule.py").is_file())
                    if model == "worldplay":
                        arms = json.loads(Path(argv[argv.index("--arms") + 1]).read_text())
                        record = next(item for item in arms if item["name"] == arm)
                        self.assertTrue(record["rgc"] and record["replan"])


    def test_bundle_has_no_model_assets_or_machine_paths(self):
        forbidden = {".pt", ".pth", ".ckpt", ".safetensors", ".mp4", ".npz", ".npy"}
        for p in BUNDLE.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                self.assertNotIn(p.suffix, forbidden)
                self.assertNotIn("/home/hj", p.read_text(), str(p))


if __name__ == "__main__":
    unittest.main()
