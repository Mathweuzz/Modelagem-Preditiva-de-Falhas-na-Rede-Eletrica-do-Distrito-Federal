"""Espelha na terceira entrega a tabela canônica de correlações do projeto."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DELIVERY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_CORRELATIONS = PROJECT_ROOT / "Fonte" / "data" / "correlacoes_consolidadas.csv"
OUTPUT = DELIVERY_ROOT / "dados" / "correlacoes_consolidadas.csv"


def main() -> None:
    if not CANONICAL_CORRELATIONS.exists():
        raise FileNotFoundError(f"Arquivo canônico ausente: {CANONICAL_CORRELATIONS}")

    frame = pd.read_csv(CANONICAL_CORRELATIONS)
    required = {"nivel_temporal", "variavel", "pearson_r"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

    output = frame.rename(columns={"variavel": "variavel_y"}).copy()
    output.insert(1, "variavel_x", "interrupcoes")
    columns = ["nivel_temporal", "variavel_x", "variavel_y", "pearson_r"]
    output = output[columns].sort_values(
        ["nivel_temporal", "pearson_r"], ascending=[True, False]
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False)
    print(f"[OK] {len(output)} correlações canônicas salvas em {OUTPUT}")


if __name__ == "__main__":
    main()
