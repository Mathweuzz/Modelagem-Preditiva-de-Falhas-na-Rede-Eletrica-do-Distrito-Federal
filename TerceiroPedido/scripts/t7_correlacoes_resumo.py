import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/home/mateus/CLEAR DATA/TerceiroPedido/TerceiroPedido")
DADOS_DIR = ROOT / "dados"
OUT_CSV = DADOS_DIR / "correlacoes_resumo.csv"

# Entradas (TerceiroPedido/dados)
BASE_DIARIA = DADOS_DIR / "base_diaria_interrupcoes_clima.csv"
SEMANAL_PRECIP = DADOS_DIR / "aggregados_semanal_interrupcoes_precipitacao.csv"
MENSAL_PRECIP = DADOS_DIR / "aggregados_mensal_interrupcoes_precipitacao.csv"
BASE_MENSAL = DADOS_DIR / "base_mensal_interrupcoes_clima_consumo.csv"


def corr(df: pd.DataFrame, a: str, b: str) -> float:
    if a not in df.columns or b not in df.columns:
        return float("nan")
    return float(df[a].corr(df[b]))


def add_row(rows, nivel, var_x, var_y, valor):
    rows.append({
        "nivel_temporal": nivel,
        "variavel_x": var_x,
        "variavel_y": var_y,
        "pearson_r": valor
    })


def main():
    rows = []

    # =====================================================
    # Diário (base integrada)
    # =====================================================
    if not BASE_DIARIA.exists():
        raise FileNotFoundError(f"Faltou: {BASE_DIARIA} (copie para TerceiroPedido/dados)")

    df_d = pd.read_csv(BASE_DIARIA)
    # correlações diárias principais
    add_row(rows, "diario", "interrupcoes", "temperatura_media", corr(df_d, "interrupcoes", "temperatura_media"))
    add_row(rows, "diario", "interrupcoes", "precipitacao_total_mm", corr(df_d, "interrupcoes", "precipitacao_total_mm"))

    # =====================================================
    # Semanal (agregado precipitação)
    # =====================================================
    if not SEMANAL_PRECIP.exists():
        raise FileNotFoundError(f"Faltou: {SEMANAL_PRECIP} (copie para TerceiroPedido/dados)")

    df_w = pd.read_csv(SEMANAL_PRECIP)
    add_row(rows, "semanal", "interrupcoes", "precipitacao_total_mm", corr(df_w, "interrupcoes", "precipitacao_total_mm"))

    # =====================================================
    # Mensal (agregado precipitação)
    # =====================================================
    if not MENSAL_PRECIP.exists():
        raise FileNotFoundError(f"Faltou: {MENSAL_PRECIP} (copie para TerceiroPedido/dados)")

    df_mprec = pd.read_csv(MENSAL_PRECIP)
    add_row(rows, "mensal", "interrupcoes", "precipitacao_total_mm", corr(df_mprec, "interrupcoes", "precipitacao_total_mm"))

    # =====================================================
    # Mensal (base completa: interrupções + clima + consumo)
    # =====================================================
    if not BASE_MENSAL.exists():
        raise FileNotFoundError(f"Faltou: {BASE_MENSAL} (copie para TerceiroPedido/dados)")

    df_m = pd.read_csv(BASE_MENSAL)
    add_row(rows, "mensal", "interrupcoes", "consumo_total_kwh", corr(df_m, "interrupcoes", "consumo_total_kwh"))
    add_row(rows, "mensal", "consumo_total_kwh", "temperatura_media", corr(df_m, "consumo_total_kwh", "temperatura_media"))
    add_row(rows, "mensal", "interrupcoes", "temperatura_media", corr(df_m, "interrupcoes", "temperatura_media"))

    # =====================================================
    # Salvar
    # =====================================================
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    print("[OK] T7 concluído: correlações salvas em:", OUT_CSV)
    print(out.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERRO]", e)
        sys.exit(1)
