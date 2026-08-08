"""Executa o pipeline principal em área temporária e promove somente após sucesso.

O escopo deste executor cobre as bases, análises e modelos mantidos em
``Fonte/src``. Figuras produzidas pelos diretórios históricos ``SegundoPedido``
e ``TerceiroPedido`` não fazem parte desta execução.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


FONTE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FONTE_DIR.parent
SRC_DIR = FONTE_DIR / "src"
TESTS_DIR = FONTE_DIR / "tests"
INTERFACE_DIR = FONTE_DIR / "interface"
DATA_DIR = FONTE_DIR / "data"
RESULTS_DIR = FONTE_DIR / "results"
MONOGRAPH_IMAGES = PROJECT_ROOT / "Monografia" / "img"

GENERATED_DATA_FILES = (
    "base_diaria_interrupcoes_clima_vento.csv",
    "vento_diario_brasilia.csv",
    "dataset_engenharia_features.csv",
    "manifesto_fontes_padronizadas.json",
    "agregados_semanais_canonicos.csv",
    "agregados_mensais_canonicos.csv",
    "correlacoes_consolidadas.csv",
)

REQUIRED_EDA_FILES = (
    "autocorrelacao_interrupcoes.png",
    "correlacao_kendall.png",
    "correlacao_spearman.png",
    "correlacoes_escala_temporal.png",
    "cross_corr_chuva_interrupcoes.png",
    "cross_corr_vento_interrupcoes.png",
    "decomposicao_interrupcoes.png",
    "decomposicao_precipitacao.png",
    "distribuicao_interrupcoes.png",
    "eda_boxplot_sazonalidade.png",
    "eda_heatmap_pearson.png",
    "eda_scatter_ventos.png",
    "eda_violin_anomalias.png",
    "evolucao_anual_interrupcoes.png",
    "serie_temporal_completa.png",
)

REQUIRED_ML_FILES = (
    "feature_importance_xgboost.png",
    "kde_residuos_modelos.png",
    "learning_curve_gru_bidirecional.png",
    "learning_curve_lstm_bidirecional.png",
    "metrics_gru_bi.csv",
    "metrics_lstm_bi.csv",
    "metrics_multihorizon.csv",
    "metrics_multihorizon_monthly.csv",
    "metrics_persistence.csv",
    "metrics_severity.csv",
    "metrics_xgboost.csv",
    "predictions_all.csv",
    "predictions_gru_bi.csv",
    "predictions_lstm_bi.csv",
    "predictions_xgboost.csv",
    "previsao_multihorizonte_degradacao.png",
    "previsao_multihorizonte_degradacao_rmse.png",
    "previsao_multihorizonte_mae_mensal.png",
    "previsao_multihorizonte_mae_rmse.png",
    "previsao_multihorizonte_metricas.csv",
    "previsao_multihorizonte_temporal_bigru.png",
    "previsao_multihorizonte_temporal_bilstm.png",
    "previsao_multihorizonte_temporal_xgboost.png",
    "scatter_heteroscedasticity.png",
    "scatter_pred_gru_bi.png",
    "scatter_pred_lstm_bi.png",
    "scatter_pred_xgboost.png",
    "ts_pred_gru_bi.png",
    "ts_pred_lstm_bi.png",
    "ts_pred_xgboost.png",
    "xgboost_best_params.json",
    "xgboost_cv_results.csv",
)


def run(script: Path, project_root: Path, *arguments: str) -> None:
    relative = script.relative_to(project_root)
    print(f"\n=== {relative} {' '.join(arguments)} ===", flush=True)
    subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=script.parent,
        check=True,
    )


def run_tests(staged_fonte: Path, staged_project: Path) -> None:
    print("\n=== Fonte/tests ===", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(staged_fonte / "tests"),
            "-p",
            "test_*.py",
            "-v",
        ],
        cwd=staged_project,
        check=True,
    )


def validate_inputs(interruptions: Path, inmet_dir: Path) -> None:
    if not interruptions.is_file():
        raise FileNotFoundError(f"CSV da ANEEL não encontrado: {interruptions}")
    if not inmet_dir.is_dir():
        raise NotADirectoryError(f"Diretório do INMET não encontrado: {inmet_dir}")


def ensure_clean_worktree() -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        raise RuntimeError(
            "O pipeline exige uma árvore de trabalho limpa para que os artefatos "
            "promovidos possam ser auditados com segurança."
        )


def prepare_workspace(stage_root: Path) -> tuple[Path, Path]:
    staged_project = stage_root / "project"
    staged_fonte = staged_project / "Fonte"
    staged_fonte.mkdir(parents=True)
    shutil.copy2(FONTE_DIR / "run_pipeline.py", staged_fonte / "run_pipeline.py")
    shutil.copytree(SRC_DIR, staged_fonte / "src")
    shutil.copytree(TESTS_DIR, staged_fonte / "tests")
    shutil.copytree(INTERFACE_DIR, staged_fonte / "interface")
    shutil.copytree(DATA_DIR, staged_fonte / "data")
    for filename in GENERATED_DATA_FILES:
        path = staged_fonte / "data" / filename
        if path.exists():
            path.unlink()

    (staged_fonte / "results" / "eda").mkdir(parents=True)
    (staged_fonte / "results" / "ml").mkdir(parents=True)
    shutil.copytree(MONOGRAPH_IMAGES, staged_project / "Monografia" / "img")
    return staged_project, staged_fonte


def validate_outputs(staged_fonte: Path) -> None:
    required = [staged_fonte / "data" / name for name in GENERATED_DATA_FILES]
    required.extend(
        staged_fonte / "results" / "eda" / name for name in REQUIRED_EDA_FILES
    )
    required.extend(
        staged_fonte / "results" / "ml" / name for name in REQUIRED_ML_FILES
    )
    missing = [path for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise RuntimeError(f"Artefatos obrigatórios ausentes ou vazios:\n{formatted}")


def synchronize_figures(staged_project: Path, staged_fonte: Path) -> int:
    destination = staged_project / "Monografia" / "img"
    copied = 0
    for group in ("eda", "ml"):
        for image in sorted((staged_fonte / "results" / group).glob("*.png")):
            shutil.copy2(image, destination / image.name)
            copied += 1
    return copied


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def promote_directories(
    pairs: list[tuple[Path, Path]],
    backup_root: Path,
    *,
    _fail_after: int | None = None,
) -> None:
    """Promove diretórios como uma transação e restaura todos em caso de falha."""
    for staged, destination in pairs:
        if not staged.is_dir():
            raise FileNotFoundError(f"Diretório de promoção ausente: {staged}")
        if staged.stat().st_dev != destination.parent.stat().st_dev:
            raise OSError(f"Promoção não atômica entre sistemas de arquivos: {staged}")

    backup_root.mkdir(parents=True, exist_ok=False)
    promoted: list[tuple[Path, Path, Path]] = []
    try:
        for index, (staged, destination) in enumerate(pairs):
            backup = backup_root / f"{index:02d}-{destination.name}"
            if destination.exists():
                destination.replace(backup)
            try:
                staged.replace(destination)
            except BaseException:
                if backup.exists():
                    backup.replace(destination)
                raise
            promoted.append((staged, destination, backup))
            if _fail_after is not None and len(promoted) == _fail_after:
                raise RuntimeError("Falha simulada após promoção parcial.")
    except BaseException:
        for staged, destination, backup in reversed(promoted):
            _remove_path(staged)
            if destination.exists():
                destination.replace(staged)
            if backup.exists():
                backup.replace(destination)
        raise
    else:
        _remove_path(backup_root)


def execute_pipeline(staged_project: Path, staged_fonte: Path, args: argparse.Namespace) -> None:
    staged_src = staged_fonte / "src"
    models = staged_src / "models"
    build_args = [
        "--interruptions",
        str(args.interruptions.resolve()),
        "--inmet-dir",
        str(args.inmet_dir.resolve()),
        "--output-dir",
        str(staged_fonte / "data"),
    ]

    run(staged_src / "build_base_from_standardized.py", staged_project, *build_args)
    run(staged_src / "03_feature_engineering.py", staged_project)
    run(staged_src / "01_eda_sazonalidade.py", staged_project)
    run(staged_src / "02_correlacoes_nao_lineares.py", staged_project)
    run(staged_src / "04_eda_basica.py", staged_project)
    run(models / "script_exploration_pipeline.py", staged_project)
    run(staged_src / "05_correlacoes_unificadas.py", staged_project)
    run_tests(staged_fonte, staged_project)
    run(models / "baseline_persistence.py", staged_project)
    run(models / "baseline_xgboost.py", staged_project)
    run(models / "lstm_bidirecional.py", staged_project)
    run(models / "gru_avancada.py", staged_project)
    run(models / "advanced_plots.py", staged_project)
    run(models / "evaluate_severity.py", staged_project)
    run(models / "previsao_multihorizonte.py", staged_project)
    run(models / "plot_multihorizonte.py", staged_project)
    run(models / "plot_multihorizonte_temporal.py", staged_project)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interruptions", type=Path, required=True)
    parser.add_argument("--inmet-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_inputs(args.interruptions, args.inmet_dir)
    ensure_clean_worktree()
    stage_root = Path(tempfile.mkdtemp(prefix=".pipeline-", dir=PROJECT_ROOT))
    print(f"Área temporária: {stage_root}")
    try:
        staged_project, staged_fonte = prepare_workspace(stage_root)
        execute_pipeline(staged_project, staged_fonte, args)
        validate_outputs(staged_fonte)
        copied = synchronize_figures(staged_project, staged_fonte)
        promote_directories(
            [
                (staged_fonte / "data", DATA_DIR),
                (staged_fonte / "results" / "eda", RESULTS_DIR / "eda"),
                (staged_fonte / "results" / "ml", RESULTS_DIR / "ml"),
                (staged_project / "Monografia" / "img", MONOGRAPH_IMAGES),
            ],
            stage_root / "rollback",
        )
    except BaseException:
        print(
            "\n[ERRO] Nenhum resultado incompleto deve ser promovido. "
            f"A área temporária foi preservada para diagnóstico: {stage_root}",
            file=sys.stderr,
        )
        raise
    else:
        shutil.rmtree(stage_root)
        print(
            "\n[OK] Pipeline concluído e promoção atômica realizada; "
            f"{copied} figuras sincronizadas com Monografia/img."
        )


if __name__ == "__main__":
    main()
