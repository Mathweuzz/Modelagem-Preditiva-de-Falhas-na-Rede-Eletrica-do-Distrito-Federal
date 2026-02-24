import sys
from pathlib import Path
import pandas as pd

ROOT = Path("/home/mateus/CLEAR DATA/TerceiroPedido/TerceiroPedido")
DADOS = ROOT / "dados"

CORR_BASE = DADOS / "correlacoes_resumo.csv"
BASE_DIARIA_VENTO = DADOS / "base_diaria_interrupcoes_clima_vento.csv"
SEMANAL_VENTO = DADOS / "aggregados_semanal_interrupcoes_vento.csv"
MENSAL_VENTO = DADOS / "aggregados_mensal_interrupcoes_vento.csv"

OUT = DADOS / "correlacoes_consolidadas.csv"


def corr(df, a, b):
    if a not in df.columns or b not in df.columns:
        return float("nan")
    return float(df[a].corr(df[b]))


def main():
    rows = []

    # 1) Começa com o que já existe (chuva/temperatura/consumo)
    if not CORR_BASE.exists():
        raise FileNotFoundError(f"Faltou: {CORR_BASE}")
    base = pd.read_csv(CORR_BASE)
    rows.extend(base.to_dict("records"))

    # 2) Vento diário
    if not BASE_DIARIA_VENTO.exists():
        raise FileNotFoundError(f"Faltou: {BASE_DIARIA_VENTO}")
    dd = pd.read_csv(BASE_DIARIA_VENTO)

    vars_vento = [
        "vento_velocidade_media_ms",
        "vento_velocidade_max_ms",
        "vento_rajada_max_ms",
        "vento_direcao_media_gr",
        "vento_direcao_moda_gr",
    ]
    for v in vars_vento:
        rows.append({
            "nivel_temporal": "diario",
            "variavel_x": "interrupcoes",
            "variavel_y": v,
            "pearson_r": corr(dd, "interrupcoes", v)
        })

    # 3) Vento semanal
    if not SEMANAL_VENTO.exists():
        raise FileNotFoundError(f"Faltou: {SEMANAL_VENTO}")
    ww = pd.read_csv(SEMANAL_VENTO)
    for v in vars_vento:
        rows.append({
            "nivel_temporal": "semanal",
            "variavel_x": "interrupcoes",
            "variavel_y": v,
            "pearson_r": corr(ww, "interrupcoes", v)
        })

    # 4) Vento mensal
    if not MENSAL_VENTO.exists():
        raise FileNotFoundError(f"Faltou: {MENSAL_VENTO}")
    mm = pd.read_csv(MENSAL_VENTO)
    for v in vars_vento:
        rows.append({
            "nivel_temporal": "mensal",
            "variavel_x": "interrupcoes",
            "variavel_y": v,
            "pearson_r": corr(mm, "interrupcoes", v)
        })

    out = pd.DataFrame(rows)

    # organiza: primeiro por abs(correlação) desc
    out["abs_r"] = out["pearson_r"].abs()
    out = out.sort_values(["nivel_temporal", "abs_r"], ascending=[True, False]).drop(columns=["abs_r"])

    out.to_csv(OUT, index=False)
    print("[OK] Salvo:", OUT)
    print(out.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERRO]", e)
        sys.exit(1)
