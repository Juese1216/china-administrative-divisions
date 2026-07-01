from __future__ import annotations

import csv
import unittest
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback.
    tomllib = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectStructureTest(unittest.TestCase):
    def test_expected_top_level_files_exist(self) -> None:
        expected = [
            "README.md",
            "requirements.txt",
            "pyproject.toml",
            "Makefile",
            "LICENSE",
            ".gitignore",
            "src/ner.py",
            "src/relation.py",
            "src/classify.py",
            "src/graph.py",
            "src/forecast.py",
            "src/rural_atlas/cli.py",
            "web/web_app.py",
            "web/webdb.py",
            "docs/project_structure.md",
        ]
        missing = [path for path in expected if not (PROJECT_ROOT / path).exists()]
        self.assertEqual([], missing)

    @unittest.skipIf(tomllib is None, "tomllib is available in Python 3.11+")
    def test_console_script_targets_exist(self) -> None:
        with (PROJECT_ROOT / "pyproject.toml").open("rb") as file:
            pyproject = tomllib.load(file)
        scripts = pyproject["project"]["scripts"]
        for command, target in scripts.items():
            module_name, function_name = target.split(":")
            module_path = PROJECT_ROOT / "src" / Path(*module_name.split(".")).with_suffix(".py")
            self.assertTrue(module_path.exists(), f"{command} -> {module_path}")
            source = module_path.read_text(encoding="utf-8")
            self.assertIn(f"def {function_name}", source, command)

    def test_generated_outputs_are_ignored(self) -> None:
        gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        required_patterns = ["data/processed/", "data/app/", "models/", "*.sqlite", "*.log"]
        for pattern in required_patterns:
            self.assertIn(pattern, gitignore)

    def test_mit_license_metadata_exists(self) -> None:
        license_text = (PROJECT_ROOT / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("MIT License", license_text)
        self.assertIn("Copyright (c) 2026 Juese1216", license_text)
        pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('license = { file = "LICENSE" }', pyproject)

    def test_mca_manifest_files_exist(self) -> None:
        manifest = PROJECT_ROOT / "data/source/mca_changes/manifest.csv"
        missing: list[str] = []
        with manifest.open(encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                filename = row.get("filename", "")
                if filename and not (manifest.parent / filename).exists():
                    missing.append(filename)
        self.assertEqual([], missing)

    def test_no_ds_store_files(self) -> None:
        ds_store_files = [
            str(path.relative_to(PROJECT_ROOT))
            for path in PROJECT_ROOT.rglob(".DS_Store")
            if ".git" not in path.parts
        ]
        self.assertEqual([], ds_store_files)


if __name__ == "__main__":
    unittest.main()
