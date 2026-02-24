import sys
from pathlib import Path
import pandas as pd

ROOT = Path("/home/mateus/CLEAR DATA/TerceiroPedido/TerceiroPedido")
DADOS = ROOT / "dados"
GRAF = ROOT / "graficos"

OUT_MD = ROOT / "TerceiroPedido.md"

CORR_CONSOL = DADOS / "correlacoes_consolidadas.csv"
CORR_VENTO = DADOS / "correlacoes_vento.csv"

BASE_MM = DADOS / "base_diaria_interrupcoes_clima_mm.csv"
BASE_DIARIA_CLIMA = DADOS / "base_diaria_interrupcoes_clima.csv"
BASE_DIARIA_VENTO = DADOS / "base_diaria_interrupcoes_clima_vento.csv"
BASE_MENSAL = DADOS / "base_mensal_interrupcoes_clima_consumo.csv"

PREV = DADOS / "previsoes_diarias_baselines.csv"

AGG_W_TEMP = DADOS / "aggregados_semanal_interrupcoes_temperatura.csv"
AGG_W_PREC = DADOS / "aggregados_semanal_interrupcoes_precipitacao.csv"
AGG_M_PREC = DADOS / "aggregados_mensal_interrupcoes_precipitacao.csv"
AGG_W_VENTO = DADOS / "aggregados_semanal_interrupcoes_vento.csv"
AGG_M_VENTO = DADOS / "aggregados_mensal_interrupcoes_vento.csv"


def md_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    df2 = df.copy()
    if len(df2) > max_rows:
        df2 = df2.head(max_rows)
    return df2.to_markdown(index=False)


def file_list_md(folder: Path) -> str:
    if not folder.exists():
        return "_(pasta não encontrada)_"
    files = sorted([p.name for p in folder.glob("*.png")])
    if not files:
        return "_(sem imagens)_"
    return "\n".join([f"- `{f}`" for f in files])


def resumo_periodo(df: pd.DataFrame, col_data: str) -> str:
    if col_data not in df.columns:
        return "N/A"
    dmin = pd.to_datetime(df[col_data]).min().date()
    dmax = pd.to_datetime(df[col_data]).max().date()
    return f"{dmin} a {dmax}"


def main():
    lines = []
    lines.append("# Terceira Entrega — Ajustes Visuais, Correlações e Vento (INMET)\n")
    lines.append("Este documento resume as alterações e resultados produzidos na **Terceira Entrega**, conforme sugestões do Prof. Jan Corrêa.\n")

    # =====================================================
    # Principais mudanças (texto pronto pro e-mail)
    # =====================================================
    lines.append("## Principais ajustes solicitados e implementados\n")
    lines.append("- **Médias móveis**: gráficos diários passaram a ser gerados em **janelas de 1 ano** (melhor legibilidade).\n"
                 "- **Padronização de cores**: interrupções em **vermelho**, temperatura em **azul** e precipitação em **azul forte**.\n"
                 "- **Interrupções x temperatura (semanal)**: geração por **intervalos anuais**.\n"
                 "- **Scatters mensais**: cada ponto representa **1 mês**, com **cor por ano/gradiente temporal** e **linha de regressão + R²**.\n"
                 "- **Previsão (baselines)**: visualização do período de teste em **janelas de 1 ano**.\n"
                 "- **Consumo**: visualização também em **GWh** para evitar notação científica no eixo e facilitar interpretação.\n"
                 "- **Vento (INMET)**: criação de variáveis diárias (velocidade, rajada e direção), integração com interrupções e análise em escalas diária/semanal/mensal.\n")

    # =====================================================
    # Dados e períodos
    # =====================================================
    lines.append("## Bases utilizadas nesta entrega\n")

    if BASE_DIARIA_CLIMA.exists():
        df = pd.read_csv(BASE_DIARIA_CLIMA)
        lines.append(f"- `dados/base_diaria_interrupcoes_clima.csv` — período: **{resumo_periodo(df,'data')}**; linhas: **{len(df)}**")
    else:
        lines.append("- `dados/base_diaria_interrupcoes_clima.csv` — _(não encontrado)_")

    if BASE_DIARIA_VENTO.exists():
        df = pd.read_csv(BASE_DIARIA_VENTO)
        lines.append(f"- `dados/base_diaria_interrupcoes_clima_vento.csv` — período: **{resumo_periodo(df,'data')}**; linhas: **{len(df)}**")
    else:
        lines.append("- `dados/base_diaria_interrupcoes_clima_vento.csv` — _(não encontrado)_")

    if BASE_MENSAL.exists():
        df = pd.read_csv(BASE_MENSAL)
        lines.append(f"- `dados/base_mensal_interrupcoes_clima_consumo.csv` — período: **{resumo_periodo(df,'data_referencia')}**; meses: **{len(df)}**")
    else:
        lines.append("- `dados/base_mensal_interrupcoes_clima_consumo.csv` — _(não encontrado)_")

    if PREV.exists():
        df = pd.read_csv(PREV)
        lines.append(f"- `dados/previsoes_diarias_baselines.csv` — período: **{resumo_periodo(df,'data')}**; linhas: **{len(df)}**")
    else:
        lines.append("- `dados/previsoes_diarias_baselines.csv` — _(não encontrado)_")

    lines.append("")

    # =====================================================
    # Resultados por tarefa (onde estão os gráficos)
    # =====================================================
    lines.append("## Artefatos gerados por tarefa\n")

    # T1
    lines.append("### T1 — Médias móveis diárias por ano (1 ano por gráfico)\n")
    lines.append(f"**Pasta:** `graficos/T1_mm_1ano/`\n\n{file_list_md(GRAF / 'T1_mm_1ano')}\n")

    # T3
    lines.append("### T3 — Interrupções x Temperatura (semanal) por ano\n")
    lines.append(f"**Pasta:** `graficos/T3_semanal_temp_ano/`\n\n{file_list_md(GRAF / 'T3_semanal_temp_ano')}\n")

    # T4
    lines.append("### T4 — Interrupções x Precipitação (semanal e mensal) com cores padronizadas\n")
    lines.append(f"**Pasta:** `graficos/T4_precipitacao/`\n\n{file_list_md(GRAF / 'T4_precipitacao')}\n")

    # T5
    lines.append("### T5 — Scatters mensais com cor por ano/gradiente + regressão\n")
    lines.append(f"**Pasta:** `graficos/T5_scatter_regressao/`\n\n{file_list_md(GRAF / 'T5_scatter_regressao')}\n")

    # T6
    lines.append("### T6 — Previsão (baselines) no teste com zoom de 1 ano\n")
    lines.append(f"**Pasta:** `graficos/T6_previsao_zoom_1ano/`\n\n{file_list_md(GRAF / 'T6_previsao_zoom_1ano')}\n")

    # T8
    lines.append("### T8 — Vento diário (INMET) integrado e gráficos diários\n")
    lines.append(f"**Pasta:** `graficos/T8_vento/`\n\n{file_list_md(GRAF / 'T8_vento')}\n")

    # T9
    lines.append("### T9 — Vento agregado semanal/mensal e gráficos\n")
    lines.append(f"**Pasta:** `graficos/T9_vento_agregados/`\n\n{file_list_md(GRAF / 'T9_vento_agregados')}\n")

    # =====================================================
    # Correlações consolidadas
    # =====================================================
    lines.append("## Correlações (Pearson) — resumo consolidado\n")

    if CORR_CONSOL.exists():
        corr = pd.read_csv(CORR_CONSOL)

        # Destaques
        lines.append("### Destaques (maiores correlações em módulo)\n")
        corr2 = corr.copy()
        corr2["abs_r"] = corr2["pearson_r"].abs()
        top = corr2.sort_values("abs_r", ascending=False).drop(columns=["abs_r"]).head(12)
        lines.append(md_table(top, max_rows=12))
        lines.append("")

        # Tabela completa
        lines.append("### Tabela completa\n")
        lines.append(md_table(corr, max_rows=200))
        lines.append("")
    else:
        lines.append("_Arquivo `correlacoes_consolidadas.csv` não encontrado._\n")

    # =====================================================
    # Interpretação curta (para e-mail / relatório)
    # =====================================================
    lines.append("## Interpretação resumida dos achados\n")
    lines.append("- **Agregação aumenta clareza do padrão**: a relação entre interrupções e variáveis meteorológicas tende a ficar mais forte em escalas **semanal/mensal** do que no diário.\n"
                 "- **Precipitação**: correlação cresce de **diário (~0,35)** para **semanal (~0,48)** e **mensal (~0,54)**.\n"
                 "- **Consumo e temperatura (mensal)**: correlação moderada/alta (**~0,56**), e interrupções também se correlacionam com consumo (**~0,48**).\n"
                 "- **Vento**: após limpeza de valores inválidos do INMET, a **direção média do vento** apresenta correlação relevante com interrupções, especialmente em escala **mensal (~0,59)**; semanal também é significativa (**~0,50**).\n"
                 "- **Observação metodológica**: direção do vento é uma variável circular (0–360°); a média simples é uma aproximação inicial e pode ser refinada com estatística circular, se necessário.\n")

    # =====================================================
    # Próximos passos (o que o prof sugeriu)
    # =====================================================
    lines.append("## Próximos passos sugeridos (planejamento)\n")
    lines.append("- Implementar modelos mais avançados de previsão (DeepAR, TFT, LSTM, GRU), garantindo **divisão temporal** sem vazamento (treino até t e previsão para t+1).\n"
                 "- Definir janela de features (lags) e estratégia de validação (walk-forward).\n"
                 "- Comparar modelos por MAE/RMSE e incluir gráfico de previsão com zoom (1 ano) no período de teste.\n")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("[OK] Markdown gerado em:", OUT_MD)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n[ERRO]", e)
        sys.exit(1)
