"""Reconstrói a base diária a partir de fontes oficiais já padronizadas.

Este módulo não lê diretamente o formato original distribuído pelos portais da
ANEEL e do INMET. As entradas precisam ter os nomes de colunas definidos abaixo.
O alvo é o total diário de interrupções únicas, sem filtro por causa. A direção
do vento é agregada com estatística circular e exposta por seno e cosseno.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_INTERRUPTION_CSV = (
    DEFAULT_INPUT_ROOT / "interrupcoes-aneel" / "dados_completos_brasilia.csv"
)
DEFAULT_INMET_DIR = DEFAULT_INPUT_ROOT / "dados_clima-inmet_limpos"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Fonte" / "data"

START_DATE = pd.Timestamp("2017-01-01")
END_DATE = pd.Timestamp("2025-05-31")

EVENT_ID_COLUMN = "NumOrdemInterrupcao"
START_COLUMN = "DatInicioInterrupcao"
HOURLY_KEY = ["data", "hora_utc"]
INMET_COLUMNS = [
    "data",
    "hora_utc",
    "precipitacao_total_horario_mm",
    "temperatura_ar_bulbo_seco_c",
    "vento_direcao_horaria_gr",
    "vento_rajada_max_ms",
    "vento_velocidade_horaria_ms",
]
INMET_VALUE_COLUMNS = INMET_COLUMNS[2:]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def circular_mean_degrees(values: pd.Series | np.ndarray) -> float:
    """Calcula a média circular em graus no intervalo [0, 360)."""
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if numeric.empty:
        return float("nan")

    radians = np.deg2rad(np.mod(numeric.to_numpy(dtype=float), 360.0))
    mean_sin = np.sin(radians).mean()
    mean_cos = np.cos(radians).mean()
    if np.hypot(mean_sin, mean_cos) < 1e-12:
        return float("nan")
    return float(np.degrees(np.arctan2(mean_sin, mean_cos)) % 360.0)


def load_unique_interruption_events(path: Path) -> tuple[pd.DataFrame, int]:
    required = [EVENT_ID_COLUMN, START_COLUMN]
    chunks: list[pd.DataFrame] = []
    raw_rows = 0
    for chunk in pd.read_csv(path, usecols=required, dtype=str, chunksize=250_000):
        raw_rows += len(chunk)
        chunks.append(chunk.drop_duplicates(EVENT_ID_COLUMN))

    if not chunks:
        raise ValueError("A fonte ANEEL não contém registros.")

    events = pd.concat(chunks, ignore_index=True).drop_duplicates(EVENT_ID_COLUMN)
    events[START_COLUMN] = pd.to_datetime(events[START_COLUMN], errors="coerce")
    if events[EVENT_ID_COLUMN].isna().any() or events[START_COLUMN].isna().any():
        raise ValueError(
            "A fonte ANEEL contém identificadores nulos ou datas iniciais inválidas."
        )

    events = events.loc[
        events[START_COLUMN].between(
            START_DATE,
            END_DATE + pd.Timedelta(days=1),
            inclusive="left",
        )
    ].copy()
    return events, raw_rows


def build_daily_target(
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Conta toda interrupção única por dia, sem filtrar a causa."""
    events = events.copy()
    events["data"] = events[START_COLUMN].dt.floor("D")
    daily = (
        events.groupby("data")[EVENT_ID_COLUMN]
        .nunique()
        .rename("interrupcoes")
        .reindex(pd.date_range(START_DATE, END_DATE, freq="D"), fill_value=0)
        .rename_axis("data")
        .reset_index()
    )
    daily["interrupcoes"] = daily["interrupcoes"].astype(int)

    summary: dict[str, int | float] = {
        "eventos_unicos": int(len(events)),
        "dias": int(len(daily)),
        "dias_sem_interrupcao": int((daily["interrupcoes"] == 0).sum()),
        "media_diaria": float(daily["interrupcoes"].mean()),
        "mediana_diaria": float(daily["interrupcoes"].median()),
        "maximo_diario": int(daily["interrupcoes"].max()),
    }
    return daily, summary


def find_inmet_a001_files(directory: Path) -> list[Path]:
    """Seleciona exatamente um arquivo da estação A001 para cada ano."""
    candidates = sorted(directory.rglob("*A001_BRASILIA*.CSV"))
    by_year: dict[int, list[Path]] = {}
    for path in candidates:
        match = re.search(r"_\d{2}-\d{2}-(20\d{2})_A_", path.name)
        if match and path.parent.name == match.group(1):
            by_year.setdefault(int(match.group(1)), []).append(path)

    expected = set(range(START_DATE.year, END_DATE.year + 1))
    missing = sorted(expected - set(by_year))
    duplicated = {year: paths for year, paths in by_year.items() if len(paths) != 1}
    if missing or duplicated:
        duplicated_names = {
            year: [path.name for path in paths] for year, paths in duplicated.items()
        }
        raise FileNotFoundError(
            "Arquivos INMET A001 inválidos: "
            f"anos ausentes={missing}; anos com quantidade diferente de um={duplicated_names}"
        )
    return [by_year[year][0] for year in sorted(expected)]


def _deduplicate_hourly_weather(hourly: pd.DataFrame) -> pd.DataFrame:
    """Remove cópias idênticas e recusa duplicatas horárias conflitantes."""
    duplicated = hourly.loc[hourly.duplicated(HOURLY_KEY, keep=False)]
    if duplicated.empty:
        return hourly

    conflicts = (
        duplicated.groupby(HOURLY_KEY, dropna=False)[INMET_VALUE_COLUMNS]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if conflicts.any():
        examples = [tuple(value) for value in conflicts[conflicts].index[:5].tolist()]
        raise ValueError(
            "A fonte INMET contém duplicatas conflitantes para data/hora; "
            f"exemplos={examples}"
        )
    return hourly.drop_duplicates(HOURLY_KEY, keep="first")


def build_daily_weather(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise ValueError("Nenhum arquivo INMET foi informado.")

    frames = [pd.read_csv(path, usecols=INMET_COLUMNS) for path in paths]
    hourly = pd.concat(frames, ignore_index=True)
    hourly["data"] = pd.to_datetime(hourly["data"], errors="coerce")
    if hourly[HOURLY_KEY].isna().any().any():
        raise ValueError("A fonte INMET contém data ou hora inválida.")

    for column in INMET_VALUE_COLUMNS:
        hourly[column] = pd.to_numeric(hourly[column], errors="coerce")

    hourly = hourly.loc[hourly["data"].between(START_DATE, END_DATE)].copy()
    hourly = _deduplicate_hourly_weather(hourly)

    for column in INMET_VALUE_COLUMNS:
        hourly.loc[hourly[column] <= -999, column] = np.nan
    hourly.loc[
        hourly["precipitacao_total_horario_mm"] < 0,
        "precipitacao_total_horario_mm",
    ] = np.nan
    for column in ["vento_rajada_max_ms", "vento_velocidade_horaria_ms"]:
        hourly.loc[hourly[column] < 0, column] = np.nan
    direction = "vento_direcao_horaria_gr"
    hourly.loc[~hourly[direction].between(0, 360), direction] = np.nan

    radians = np.deg2rad(hourly[direction] % 360.0)
    hourly["_dir_sin"] = np.sin(radians)
    hourly["_dir_cos"] = np.cos(radians)
    grouped = hourly.groupby("data", sort=True)
    daily = grouped.agg(
        temperatura_media=("temperatura_ar_bulbo_seco_c", "mean"),
        vento_velocidade_media_ms=("vento_velocidade_horaria_ms", "mean"),
        vento_velocidade_max_ms=("vento_velocidade_horaria_ms", "max"),
        vento_rajada_max_ms=("vento_rajada_max_ms", "max"),
        _dir_sin=("_dir_sin", "mean"),
        _dir_cos=("_dir_cos", "mean"),
        n_registros=("vento_velocidade_horaria_ms", "count"),
    )
    daily["precipitacao_total_mm"] = grouped[
        "precipitacao_total_horario_mm"
    ].sum(min_count=1)

    norm = np.hypot(daily["_dir_sin"], daily["_dir_cos"])
    valid_direction = norm >= 1e-12
    daily["vento_dir_sin"] = np.nan
    daily["vento_dir_cos"] = np.nan
    daily.loc[valid_direction, "vento_dir_sin"] = (
        daily.loc[valid_direction, "_dir_sin"] / norm.loc[valid_direction]
    )
    daily.loc[valid_direction, "vento_dir_cos"] = (
        daily.loc[valid_direction, "_dir_cos"] / norm.loc[valid_direction]
    )
    daily["vento_direcao_media_circular_gr"] = (
        np.degrees(np.arctan2(daily["vento_dir_sin"], daily["vento_dir_cos"]))
        % 360.0
    )
    daily = daily.drop(columns=["_dir_sin", "_dir_cos"]).reset_index()
    return daily[
        [
            "data",
            "temperatura_media",
            "precipitacao_total_mm",
            "vento_velocidade_media_ms",
            "vento_velocidade_max_ms",
            "vento_rajada_max_ms",
            "vento_direcao_media_circular_gr",
            "vento_dir_sin",
            "vento_dir_cos",
            "n_registros",
        ]
    ]


def build_base(
    interruption_csv: Path,
    inmet_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    events, raw_rows = load_unique_interruption_events(interruption_csv)
    target, target_summary = build_daily_target(events)
    inmet_files = find_inmet_a001_files(inmet_dir)
    weather = build_daily_weather(inmet_files)

    base = target.merge(weather, on="data", how="left", validate="one_to_one")
    expected_days = (END_DATE - START_DATE).days + 1
    if len(base) != expected_days:
        raise ValueError(f"Esperados {expected_days} dias; obtidos {len(base)}.")

    weather_path = output_dir / "vento_diario_brasilia.csv"
    base_path = output_dir / "base_diaria_interrupcoes_clima_vento.csv"
    manifest_path = output_dir / "manifesto_fontes_padronizadas.json"
    weather.to_csv(weather_path, index=False)
    base.drop(columns=["vento_direcao_media_circular_gr"]).to_csv(
        base_path,
        index=False,
    )

    manifest: dict[str, object] = {
        "formato_entrada": "arquivos oficiais previamente padronizados",
        "periodo": {"inicio": str(START_DATE.date()), "fim": str(END_DATE.date())},
        "alvo": {
            "definicao": "total diario de interrupcoes unicas da ANEEL",
            "filtro_por_causa": False,
            "deduplicacao": EVENT_ID_COLUMN,
        },
        "interrupcoes": {
            "arquivo": interruption_csv.name,
            "sha256": sha256(interruption_csv),
            "linhas_entrada": raw_rows,
            **target_summary,
        },
        "inmet": [
            {"arquivo": path.name, "sha256": sha256(path)} for path in inmet_files
        ],
        "saidas": [base_path.name, weather_path.name],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interruptions", type=Path, default=DEFAULT_INTERRUPTION_CSV)
    parser.add_argument("--inmet-dir", type=Path, default=DEFAULT_INMET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = build_base(arguments.interruptions, arguments.inmet_dir, arguments.output_dir)
    stats = result["interrupcoes"]
    print(
        "[OK] Alvo total diário (sem filtro por causa): "
        f"{stats['eventos_unicos']:,} eventos únicos; "
        f"média diária={stats['media_diaria']:.2f}; máximo={stats['maximo_diario']}"
    )
