"""Faixas descritivas do volume diário de interrupções.

Os limites foram definidos pelos autores para análise exploratória. Eles não
representam categorias regulatórias oficiais da ANEEL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


LOW_VOLUME_LABEL = "Baixo volume (<200)"
MID_VOLUME_LABEL = "Volume intermediário (200–400)"
HIGH_VOLUME_LABEL = "Alto volume (>400)"
VOLUME_BAND_LABELS = (LOW_VOLUME_LABEL, MID_VOLUME_LABEL, HIGH_VOLUME_LABEL)


def volume_band_masks(target: pd.Series) -> dict[str, pd.Series]:
    """Cria uma única máscara por faixa sobre a série-alvo recebida."""
    values = pd.to_numeric(target, errors="coerce")
    if values.isna().any():
        raise ValueError("O alvo contém valores ausentes ou não numéricos.")
    return {
        LOW_VOLUME_LABEL: values < 200,
        MID_VOLUME_LABEL: (values >= 200) & (values <= 400),
        HIGH_VOLUME_LABEL: values > 400,
    }


def classify_volume(target: pd.Series) -> pd.Series:
    """Classifica o alvo nas mesmas faixas usadas na avaliação dos modelos."""
    masks = volume_band_masks(target)
    classified = np.select(
        [mask.to_numpy() for mask in masks.values()],
        list(masks),
        default=None,
    )
    categorical = pd.Categorical(
        classified,
        categories=VOLUME_BAND_LABELS,
        ordered=True,
    )
    return pd.Series(categorical, index=target.index, name="Faixa de volume")
