"""Funções de métricas compartilhadas pelos modelos e pela interface."""

from __future__ import annotations

import numpy as np


def mean_absolute_percentage_error(y_true, y_pred) -> float:
    """Calcula MAPE excluindo somente alvos iguais a zero.

    MAPE não é definido quando o valor real é zero. Esses casos são removidos do
    denominador, em vez de receberem um epsilon que alteraria a métrica.
    """
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError(
            f"y_true e y_pred devem ter o mesmo formato: {true.shape} != {pred.shape}"
        )

    valid = true != 0
    if not valid.any():
        return float("nan")
    return float(np.mean(np.abs((true[valid] - pred[valid]) / true[valid])) * 100)
