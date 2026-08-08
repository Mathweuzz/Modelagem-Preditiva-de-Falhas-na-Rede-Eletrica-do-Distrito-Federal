"""Resolução portátil de caminhos para os scripts históricos da segunda entrega."""

from __future__ import annotations

import os
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Localiza a raiz do repositório por marcadores, sem depender da profundidade."""
    resolved = (start or Path(__file__)).resolve()
    for candidate in [resolved.parent, *resolved.parents]:
        if (candidate / "Fonte").is_dir() and (candidate / "Monografia").is_dir():
            return candidate
    raise RuntimeError(f"Não foi possível localizar a raiz do projeto a partir de {resolved}")


PROJECT_ROOT = find_project_root()
DELIVERY_ROOT = Path(__file__).resolve().parents[1]


def raw_data_root() -> Path:
    """Retorna a raiz de dados externos, configurável por variável de ambiente."""
    configured = os.environ.get("TCC_RAW_DATA_ROOT")
    return Path(configured).expanduser().resolve() if configured else PROJECT_ROOT
