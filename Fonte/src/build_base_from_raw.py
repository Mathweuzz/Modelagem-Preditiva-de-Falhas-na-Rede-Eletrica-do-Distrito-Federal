"""Build the canonical daily modelling base from ANEEL and INMET raw files.

The target is the total number of unique interruptions per day; no generating-
fact/cause filter is applied. Wind direction is aggregated with circular
statistics and represented by sine/cosine components.
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
DEFAULT_RAW_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_INTERRUPTION_CSV = (
    DEFAULT_RAW_ROOT / "interrupcoes-aneel" / "dados_completos_brasilia.csv"
)
DEFAULT_INMET_DIR = DEFAULT_RAW_ROOT / "dados_clima-inmet_limpos"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Fonte" / "data"

START_DATE = pd.Timestamp("2017-01-01")
END_DATE = pd.Timestamp("2025-05-31")

EVENT_ID_COLUMN = "NumOrdemInterrupcao"
START_COLUMN = "DatInicioInterrupcao"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def circular_mean_degrees(values: pd.Series | np.ndarray) -> float:
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

    events = pd.concat(chunks, ignore_index=True).drop_duplicates(EVENT_ID_COLUMN)
    events[START_COLUMN] = pd.to_datetime(events[START_COLUMN], errors="coerce")
    events = events.loc[
        events[START_COLUMN].between(START_DATE, END_DATE + pd.Timedelta(days=1), inclusive="left")
    ].copy()
    if events[EVENT_ID_COLUMN].isna().any() or events[START_COLUMN].isna().any():
        raise ValueError("The ANEEL source contains null event identifiers or invalid start dates.")
    return events, raw_rows


def build_daily_target(events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Count every unique ANEEL interruption by day, without filtering by cause."""
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
    candidates = sorted(directory.rglob("*A001_BRASILIA*.CSV"))
    selected: list[Path] = []
    for path in candidates:
        match = re.search(r"_\d{2}-\d{2}-(20\d{2})_A_", path.name)
        if match and path.parent.name == match.group(1):
            selected.append(path)
    years = {int(re.search(r"_\d{2}-\d{2}-(20\d{2})_A_", p.name).group(1)) for p in selected}
    expected = set(range(START_DATE.year, END_DATE.year + 1))
    if years != expected:
        raise FileNotFoundError(
            f"INMET A001 files are incomplete: expected {sorted(expected)}, found {sorted(years)}"
        )
    return selected


def build_daily_weather(paths: list[Path]) -> pd.DataFrame:
    columns = [
        "data",
        "hora_utc",
        "precipitacao_total_horario_mm",
        "temperatura_ar_bulbo_seco_c",
        "vento_direcao_horaria_gr",
        "vento_rajada_max_ms",
        "vento_velocidade_horaria_ms",
    ]
    frames = [pd.read_csv(path, usecols=columns) for path in paths]
    hourly = pd.concat(frames, ignore_index=True)
    hourly["data"] = pd.to_datetime(hourly["data"], errors="coerce")
    hourly = hourly.loc[hourly["data"].between(START_DATE, END_DATE)].copy()
    hourly = hourly.drop_duplicates(["data", "hora_utc"])

    numeric_columns = columns[2:]
    for column in numeric_columns:
        hourly[column] = pd.to_numeric(hourly[column], errors="coerce")
        hourly.loc[hourly[column] <= -999, column] = np.nan
    hourly.loc[hourly["precipitacao_total_horario_mm"] < 0, "precipitacao_total_horario_mm"] = np.nan
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
    daily["precipitacao_total_mm"] = grouped["precipitacao_total_horario_mm"].sum(min_count=1)

    norm = np.hypot(daily["_dir_sin"], daily["_dir_cos"])
    daily["vento_dir_sin"] = daily["_dir_sin"] / norm
    daily["vento_dir_cos"] = daily["_dir_cos"] / norm
    daily["vento_direcao_media_circular_gr"] = (
        np.degrees(np.arctan2(daily["vento_dir_sin"], daily["vento_dir_cos"])) % 360.0
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


def build_base(interruption_csv: Path, inmet_dir: Path, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    events, raw_rows = load_unique_interruption_events(interruption_csv)
    target, target_summary = build_daily_target(events)
    inmet_files = find_inmet_a001_files(inmet_dir)
    weather = build_daily_weather(inmet_files)

    base = target.merge(weather, on="data", how="left", validate="one_to_one")
    expected_days = (END_DATE - START_DATE).days + 1
    if len(base) != expected_days:
        raise ValueError(f"Expected {expected_days} daily rows, obtained {len(base)}")

    weather_path = output_dir / "vento_diario_brasilia.csv"
    base_path = output_dir / "base_diaria_interrupcoes_clima_vento.csv"
    manifest_path = output_dir / "manifesto_dados_brutos.json"
    weather.to_csv(weather_path, index=False)
    base.drop(columns=["vento_direcao_media_circular_gr"]).to_csv(base_path, index=False)

    manifest: dict[str, object] = {
        "periodo": {"inicio": str(START_DATE.date()), "fim": str(END_DATE.date())},
        "alvo": {
            "definicao": "total diario de interrupcoes unicas da ANEEL",
            "filtro_por_causa": False,
            "deduplicacao": EVENT_ID_COLUMN,
        },
        "interrupcoes": {
            "arquivo": interruption_csv.name,
            "sha256": sha256(interruption_csv),
            "linhas_brutas": raw_rows,
            **target_summary,
        },
        "inmet": [
            {"arquivo": path.name, "sha256": sha256(path)} for path in inmet_files
        ],
        "saidas": [base_path.name, weather_path.name],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
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
        "[OK] Target total diario (sem filtro por causa): "
        f"{stats['eventos_unicos']:,} eventos unicos; "
        f"media diaria={stats['media_diaria']:.2f}; maximo={stats['maximo_diario']}"
    )
