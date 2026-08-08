"""Run the raw-to-results pipeline once and synchronize monograph figures."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


FONTE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FONTE_DIR.parent
SRC_DIR = FONTE_DIR / "src"
MODELS_DIR = SRC_DIR / "models"
RESULTS_DIR = FONTE_DIR / "results"
MONOGRAPH_IMAGES = PROJECT_ROOT / "Monografia" / "img"


def run(script: Path, *arguments: str) -> None:
    print(f"\n=== {script.relative_to(PROJECT_ROOT)} {' '.join(arguments)} ===", flush=True)
    subprocess.run([sys.executable, str(script), *arguments], cwd=script.parent, check=True)


def run_tests() -> None:
    print("\n=== Fonte/tests ===", flush=True)
    subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(FONTE_DIR / "tests"),
         "-p", "test_*.py", "-q"],
        cwd=PROJECT_ROOT,
        check=True,
    )


def archive_previous_results() -> None:
    existing = [RESULTS_DIR / name for name in ("eda", "ml") if (RESULTS_DIR / name).exists()]
    if not existing:
        return
    destination = RESULTS_DIR / "archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
    destination.mkdir(parents=True, exist_ok=False)
    for path in existing:
        shutil.move(str(path), destination / path.name)
    print(f"Previous results archived in {destination}")


def synchronize_figures() -> int:
    MONOGRAPH_IMAGES.mkdir(parents=True, exist_ok=True)
    copied = 0
    for result_group in (RESULTS_DIR / "eda", RESULTS_DIR / "ml"):
        for image in result_group.glob("*.png"):
            shutil.copy2(image, MONOGRAPH_IMAGES / image.name)
            copied += 1
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interruptions", type=Path)
    parser.add_argument("--inmet-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_args: list[str] = []
    if args.interruptions:
        raw_args += ["--interruptions", str(args.interruptions)]
    if args.inmet_dir:
        raw_args += ["--inmet-dir", str(args.inmet_dir)]

    archive_previous_results()
    run(SRC_DIR / "build_base_from_raw.py", *raw_args)
    run(SRC_DIR / "03_feature_engineering.py")
    run(SRC_DIR / "01_eda_sazonalidade.py")
    run(SRC_DIR / "02_correlacoes_nao_lineares.py")
    run(SRC_DIR / "04_eda_basica.py")
    run(MODELS_DIR / "script_exploration_pipeline.py")
    run(SRC_DIR / "05_correlacoes_unificadas.py")
    run_tests()
    run(MODELS_DIR / "baseline_persistence.py")
    run(MODELS_DIR / "baseline_xgboost.py")
    run(MODELS_DIR / "lstm_bidirecional.py")
    run(MODELS_DIR / "gru_avancada.py")
    run(MODELS_DIR / "advanced_plots.py")
    run(MODELS_DIR / "evaluate_severity.py")
    run(MODELS_DIR / "previsao_multihorizonte.py")
    run(MODELS_DIR / "plot_multihorizonte.py")
    run(MODELS_DIR / "plot_multihorizonte_temporal.py")
    copied = synchronize_figures()
    print(f"\n[OK] Final run completed; {copied} figures synchronized with Monografia/img.")


if __name__ == "__main__":
    main()
