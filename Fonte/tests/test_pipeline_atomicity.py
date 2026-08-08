"""Regressões para validação e promoção transacional do pipeline."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


FONTE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FONTE_DIR))

import run_pipeline  # noqa: E402


def make_directory(path: Path, filename: str, content: str) -> Path:
    path.mkdir(parents=True)
    (path / filename).write_text(content, encoding="utf-8")
    return path


class PipelineAtomicityTests(unittest.TestCase):
    def test_workspace_starts_without_previous_generated_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            src = make_directory(source / "src", "script.py", "pass")
            tests = make_directory(source / "tests", "test_example.py", "pass")
            interface = make_directory(source / "interface", "model_service.py", "pass")
            data = make_directory(source / "data", "raw.csv", "raw")
            images = make_directory(source / "images", "existing.png", "old")
            fonte_dir = make_directory(source / "Fonte", "run_pipeline.py", "pass")
            (data / "generated.csv").write_text("stale", encoding="utf-8")
            stage = root / "stage"

            with (
                patch.object(run_pipeline, "SRC_DIR", src),
                patch.object(run_pipeline, "TESTS_DIR", tests),
                patch.object(run_pipeline, "INTERFACE_DIR", interface),
                patch.object(run_pipeline, "DATA_DIR", data),
                patch.object(run_pipeline, "FONTE_DIR", fonte_dir),
                patch.object(run_pipeline, "MONOGRAPH_IMAGES", images),
                patch.object(
                    run_pipeline, "GENERATED_DATA_FILES", ("generated.csv",)
                ),
            ):
                project, fonte = run_pipeline.prepare_workspace(stage)

            self.assertEqual((fonte / "data" / "raw.csv").read_text(), "raw")
            self.assertEqual((fonte / "run_pipeline.py").read_text(), "pass")
            self.assertEqual(
                (fonte / "interface" / "model_service.py").read_text(), "pass"
            )
            self.assertFalse((fonte / "data" / "generated.csv").exists())
            self.assertEqual(list((fonte / "results" / "eda").iterdir()), [])
            self.assertEqual(list((fonte / "results" / "ml").iterdir()), [])
            self.assertEqual(
                (project / "Monografia" / "img" / "existing.png").read_text(),
                "old",
            )

    def test_successful_promotion_replaces_every_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged_a = make_directory(root / "staged-a", "new.txt", "new-a")
            staged_b = make_directory(root / "staged-b", "new.txt", "new-b")
            final_a = make_directory(root / "final-a", "old.txt", "old-a")
            final_b = make_directory(root / "final-b", "old.txt", "old-b")
            backup = root / "rollback"

            run_pipeline.promote_directories(
                [(staged_a, final_a), (staged_b, final_b)],
                backup,
            )

            self.assertEqual((final_a / "new.txt").read_text(), "new-a")
            self.assertEqual((final_b / "new.txt").read_text(), "new-b")
            self.assertFalse((final_a / "old.txt").exists())
            self.assertFalse(backup.exists())

    def test_partial_failure_restores_every_official_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged_a = make_directory(root / "staged-a", "new.txt", "new-a")
            staged_b = make_directory(root / "staged-b", "new.txt", "new-b")
            final_a = make_directory(root / "final-a", "old.txt", "old-a")
            final_b = make_directory(root / "final-b", "old.txt", "old-b")

            with self.assertRaisesRegex(RuntimeError, "simulada"):
                run_pipeline.promote_directories(
                    [(staged_a, final_a), (staged_b, final_b)],
                    root / "rollback",
                    _fail_after=2,
                )

            self.assertEqual((final_a / "old.txt").read_text(), "old-a")
            self.assertEqual((final_b / "old.txt").read_text(), "old-b")
            self.assertEqual((staged_a / "new.txt").read_text(), "new-a")
            self.assertEqual((staged_b / "new.txt").read_text(), "new-b")

    def test_validation_rejects_missing_or_empty_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fonte = Path(directory) / "Fonte"
            (fonte / "data").mkdir(parents=True)
            (fonte / "results" / "eda").mkdir(parents=True)
            (fonte / "results" / "ml").mkdir(parents=True)
            (fonte / "data" / "base.csv").write_text("ok", encoding="utf-8")
            (fonte / "results" / "eda" / "eda.png").write_bytes(b"png")
            (fonte / "results" / "ml" / "metrics.csv").write_text(
                "", encoding="utf-8"
            )

            with (
                patch.object(run_pipeline, "GENERATED_DATA_FILES", ("base.csv",)),
                patch.object(run_pipeline, "REQUIRED_EDA_FILES", ("eda.png",)),
                patch.object(run_pipeline, "REQUIRED_ML_FILES", ("metrics.csv",)),
            ):
                with self.assertRaisesRegex(RuntimeError, "vazios"):
                    run_pipeline.validate_outputs(fonte)
                (fonte / "results" / "ml" / "metrics.csv").write_text(
                    "ok", encoding="utf-8"
                )
                run_pipeline.validate_outputs(fonte)

    def test_figure_synchronization_copies_only_png_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            fonte = project / "Fonte"
            destination = project / "Monografia" / "img"
            destination.mkdir(parents=True)
            for group in ("eda", "ml"):
                result = fonte / "results" / group
                result.mkdir(parents=True)
                (result / f"{group}.png").write_bytes(group.encode())
                (result / f"{group}.csv").write_text("ignored", encoding="utf-8")

            copied = run_pipeline.synchronize_figures(project, fonte)

            self.assertEqual(copied, 2)
            self.assertEqual((destination / "eda.png").read_bytes(), b"eda")
            self.assertFalse((destination / "eda.csv").exists())


if __name__ == "__main__":
    unittest.main()
