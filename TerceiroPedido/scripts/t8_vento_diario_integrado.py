import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("/home/mateus/CLEAR DATA/TerceiroPedido/TerceiroPedido")
DADOS_DIR = ROOT / "dados"
GRAFICOS_DIR = ROOT / "graficos" / "T8_vento"
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

# INMET horários (limpos)
INMET_DIR = Path("/home/mateus/CLEAR DATA/dados_clima-inmet_limpos")

# Base diária (com interrupções) já pronta
BASE_DIARIA = DADOS_DIR / "base_diaria_interrupcoes_clima.csv"

# Saídas
VENTO_DIARIO_CSV = DADOS_DIR / "vento_diario_brasilia.csv"
BASE_DIARIA_VENTO_CSV = DADOS_DIR / "base_diaria_interrupcoes_clima_vento.csv"
CORRELACOES_CSV = DADOS_DIR / "correlacoes_vento.csv"


def listar_csvs_inmet(inmet_dir: Path) -> list[Path]:
    arquivos = []
    for ano_dir in sorted(inmet_dir.glob("*")):
        if not ano_dir.is_dir():
            continue
        for csv in sorted(ano_dir.glob("*.CSV")):
            arquivos.append(csv)
    return arquivos


def ler_inmet_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "data" not in df.columns:
        raise ValueError(f"[{path.name}] coluna 'data' não encontrada.")
    df["data"] = pd.to_datetime(df["data"])

    cols = ["vento_direcao_horaria_gr", "vento_rajada_max_ms", "vento_velocidade_horaria_ms"]
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"[{path.name}] coluna '{col}' não encontrada.")
        df[col] = pd.to_numeric(df[col], errors="coerce")

        # INMET costuma usar -9999 como "sem dado"
        df.loc[df[col] <= -999, col] = pd.NA

    # Regras físicas simples (evita distorção):
    # velocidade e rajada não podem ser negativas
    df.loc[df["vento_velocidade_horaria_ms"] < 0, "vento_velocidade_horaria_ms"] = pd.NA
    df.loc[df["vento_rajada_max_ms"] < 0, "vento_rajada_max_ms"] = pd.NA

    # direção deve estar entre 0 e 360
    df.loc[(df["vento_direcao_horaria_gr"] < 0) | (df["vento_direcao_horaria_gr"] > 360),
           "vento_direcao_horaria_gr"] = pd.NA

    return df[["data", "vento_direcao_horaria_gr", "vento_rajada_max_ms", "vento_velocidade_horaria_ms"]]



def moda_series(s: pd.Series):
    s2 = s.dropna()
    if s2.empty:
        return float("nan")
    return float(s2.round(0).mode().iloc[0])


def construir_vento_diario():
    print("[1] Lendo INMET horários (2017-2025) e agregando vento por dia...")
    arquivos = listar_csvs_inmet(INMET_DIR)
    if not arquivos:
        raise FileNotFoundError(f"Nenhum CSV INMET encontrado em: {INMET_DIR}")

    dfs = []
    for p in arquivos:
        try:
            dfs.append(ler_inmet_csv(p))
        except Exception as e:
            print(f"[WARN] Falha ao ler {p}: {e}")

    if not dfs:
        raise RuntimeError("Nenhum CSV INMET foi lido com sucesso.")

    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values("data")

    vento_diario = df.groupby("data").agg(
        vento_velocidade_media_ms=("vento_velocidade_horaria_ms", "mean"),
        vento_velocidade_max_ms=("vento_velocidade_horaria_ms", "max"),
        vento_rajada_max_ms=("vento_rajada_max_ms", "max"),
        vento_direcao_media_gr=("vento_direcao_horaria_gr", "mean"),
        vento_direcao_moda_gr=("vento_direcao_horaria_gr", moda_series),
        n_registros=("vento_velocidade_horaria_ms", "count"),
    ).reset_index()

    print(f"[1] Vento diário gerado com {len(vento_diario)} dias: {vento_diario['data'].min().date()} a {vento_diario['data'].max().date()}")
    print("[1] Salvando:", VENTO_DIARIO_CSV)
    vento_diario.to_csv(VENTO_DIARIO_CSV, index=False)
    return vento_diario


def integrar_com_interrupcoes(vento_diario: pd.DataFrame):
    print("[2] Lendo base diária de interrupções + clima:", BASE_DIARIA)
    if not BASE_DIARIA.exists():
        raise FileNotFoundError(
            f"CSV não encontrado: {BASE_DIARIA}\n"
            "Copie ele do SegundoPedido para TerceiroPedido/dados antes de rodar."
        )

    base = pd.read_csv(BASE_DIARIA)
    if "data" not in base.columns:
        raise ValueError("Coluna 'data' não encontrada em base_diaria_interrupcoes_clima.csv")

    base["data"] = pd.to_datetime(base["data"])
    vento_diario["data"] = pd.to_datetime(vento_diario["data"])

    print("[2] Integrando por data...")
    merged = base.merge(vento_diario, on="data", how="left")

    print(f"[2] Base integrada com vento: {len(merged)} linhas.")
    print("[2] Salvando:", BASE_DIARIA_VENTO_CSV)
    merged.to_csv(BASE_DIARIA_VENTO_CSV, index=False)
    return merged


def correlacoes_e_graficos(df: pd.DataFrame):
    print("[3] Calculando correlações (Pearson) interrupções x vento...")
    rows = []
    alvo = "interrupcoes"
    vars_vento = [
        "vento_velocidade_media_ms",
        "vento_velocidade_max_ms",
        "vento_rajada_max_ms",
        "vento_direcao_media_gr",
        "vento_direcao_moda_gr",
    ]

    for v in vars_vento:
        if v not in df.columns:
            continue
        r = df[alvo].corr(df[v])
        rows.append({"variavel": v, "pearson_r": r})

    out = pd.DataFrame(rows).sort_values("pearson_r", ascending=False)
    print(out.to_string(index=False))
    out.to_csv(CORRELACOES_CSV, index=False)
    print("[3] Salvando:", CORRELACOES_CSV)

    # Gráficos simples (série diária) - interrupções vs vento_velocidade_media_ms e rajada
    # (duplo eixo)
    def plot_duplo(y2col, titulo, out_png):
        sub = df.dropna(subset=[y2col]).copy()
        if sub.empty:
            print(f"[SKIP] Sem dados para {y2col}")
            return
        plt.figure(figsize=(12, 6))
        ax1 = plt.gca()
        ax1.plot(sub["data"], sub["interrupcoes"], color="red", alpha=0.6, label="Interrupções (diário)")
        ax1.set_xlabel("Data")
        ax1.set_ylabel("Interrupções", color="red")
        ax1.tick_params(axis="x", rotation=45)

        ax2 = ax1.twinx()
        ax2.plot(sub["data"], sub[y2col], color="blue", alpha=0.7, label=y2col)
        ax2.set_ylabel(y2col, color="blue")

        plt.title(titulo)
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="upper left")

        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()
        print("[OK] Gerado:", out_png)

    plot_duplo("vento_velocidade_media_ms",
               "Interrupções diárias x Velocidade média do vento (m/s)",
               GRAFICOS_DIR / "diario_interrupcoes_vs_vento_vel_media.png")

    plot_duplo("vento_rajada_max_ms",
               "Interrupções diárias x Rajada máxima do vento (m/s)",
               GRAFICOS_DIR / "diario_interrupcoes_vs_rajada_max.png")


def main():
    print("INMET_DIR =", INMET_DIR)
    print("BASE_DIARIA =", BASE_DIARIA)
    vento = construir_vento_diario()
    merged = integrar_com_interrupcoes(vento)
    correlacoes_e_graficos(merged)
    print("\n[OK] T8 concluído: vento diário + integração + correlações + gráficos.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERRO]", e)
        sys.exit(1)
