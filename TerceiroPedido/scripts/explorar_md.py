"""Regenera o resumo Markdown da terceira entrega com dados canônicos atuais."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path(__file__).resolve().parents[1]
DADOS = ROOT / "dados"
GRAF = ROOT / "graficos"
OUT_MD = ROOT / "TerceiroPedido.md"

CORR_CONSOL = PROJECT_ROOT / "Fonte" / "data" / "correlacoes_consolidadas.csv"
BASE_DIARIA_CLIMA = DADOS / "base_diaria_interrupcoes_clima.csv"
BASE_DIARIA_VENTO = DADOS / "base_diaria_interrupcoes_clima_vento.csv"
BASE_MENSAL = DADOS / "base_mensal_interrupcoes_clima_consumo.csv"
PREV = DADOS / "previsoes_diarias_baselines.csv"


def md_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    """Formata uma tabela Markdown sem depender do pacote opcional tabulate."""
    clipped = frame.head(max_rows).copy()
    headers = [str(column) for column in clipped.columns]
    rows = [
        [str(value) for value in row]
        for row in clipped.itertuples(index=False, name=None)
    ]
    widths = [
        max([len(headers[index]), *[len(row[index]) for row in rows]])
        for index in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(headers)) + " |",
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(
        "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
        for row in rows
    )
    return "\n".join(lines)


def file_list_md(folder: Path) -> str:
    if not folder.exists():
        return "_(pasta não encontrada)_"
    files = sorted(path.name for path in folder.glob("*.png"))
    if not files:
        return "_(sem imagens)_"
    return "\n".join(f"- `{name}`" for name in files)


def resumo_periodo(frame: pd.DataFrame, date_column: str) -> str:
    if date_column not in frame.columns:
        return "N/A"
    dates = pd.to_datetime(frame[date_column])
    return f"{dates.min().date()} a {dates.max().date()}"


def describe_base(lines: list[str], path: Path, date_column: str, unit: str) -> None:
    relative = path.relative_to(ROOT)
    if path.exists():
        frame = pd.read_csv(path)
        lines.append(
            f"- `{relative}` — período: **{resumo_periodo(frame, date_column)}**; "
            f"{unit}: **{len(frame)}**"
        )
    else:
        lines.append(f"- `{relative}` — _(não encontrado)_")


def main() -> None:
    lines = [
        "# Terceira Entrega — Ajustes Visuais, Correlações e Vento (INMET)\n",
        "Este documento resume as alterações e os resultados produzidos na terceira entrega.\n",
        "## Principais ajustes implementados\n",
        "- **Médias móveis**: gráficos diários em janelas de um ano, para melhorar a legibilidade.\n"
        "- **Padronização de cores**: interrupções em vermelho, temperatura em azul e precipitação em azul-escuro.\n"
        "- **Agregações**: visualizações semanais e mensais, além de dispersões mensais com regressão e $R^2$.\n"
        "- **Consumo**: visualização em GWh para facilitar a leitura dos eixos.\n"
        "- **Vento**: variáveis diárias da estação A001, com direção representada por seno e cosseno e agregada circularmente.\n",
        "## Bases utilizadas nesta entrega\n",
    ]

    describe_base(lines, BASE_DIARIA_CLIMA, "data", "linhas")
    describe_base(lines, BASE_DIARIA_VENTO, "data", "linhas")
    describe_base(lines, BASE_MENSAL, "data_referencia", "meses")
    describe_base(lines, PREV, "data", "linhas")
    lines.append("")

    lines.append("## Artefatos gerados por tarefa\n")
    tasks = [
        ("T1", "Médias móveis diárias por ano", "T1_mm_1ano"),
        ("T3", "Interrupções e temperatura semanal por ano", "T3_semanal_temp_ano"),
        ("T4", "Precipitação semanal e mensal", "T4_precipitacao"),
        ("T5", "Dispersões mensais com regressão", "T5_scatter_regressao"),
        ("T6", "Baselines no período de teste", "T6_previsao_zoom_1ano"),
        ("T8", "Vento diário integrado", "T8_vento"),
        ("T9", "Vento agregado semanal e mensal", "T9_vento_agregados"),
    ]
    for code, title, folder in tasks:
        lines.append(f"### {code} — {title}\n")
        lines.append(f"**Pasta:** `graficos/{folder}/`\n\n{file_list_md(GRAF / folder)}\n")

    lines.append("## Correlações de Pearson — resumo canônico\n")
    if CORR_CONSOL.exists():
        correlations = pd.read_csv(CORR_CONSOL)
        top = (
            correlations.assign(abs_r=correlations["pearson_r"].abs())
            .sort_values("abs_r", ascending=False)
            .drop(columns="abs_r")
            .head(12)
        )
        lines.extend([
            "### Maiores correlações em módulo\n",
            md_table(top, max_rows=12),
            "",
            "### Tabela completa\n",
            md_table(correlations, max_rows=200),
            "",
        ])
    else:
        lines.append(f"_Arquivo canônico não encontrado: `{CORR_CONSOL}`._\n")

    lines.extend([
        "## Interpretação resumida dos achados\n",
        "- A agregação temporal não fortalece todas as relações, mas evidencia alguns padrões acumulados.\n"
        "- A precipitação apresenta correlação de aproximadamente **0,348** no diário, **0,495** no semanal e **0,539** no mensal.\n"
        "- No nível mensal, interrupções e consumo têm correlação de aproximadamente **0,476**.\n"
        "- A direção do vento é circular: seus componentes seno e cosseno alcançam, no mensal, aproximadamente **-0,522** e **0,570**, respectivamente.\n"
        "- Correlação descreve associação e, isoladamente, não demonstra causalidade.\n",
        "## Próximos passos sugeridos\n",
        "- Comparar modelos preditivos por divisão temporal e validação walk-forward, sempre evitando vazamento.\n"
        "- Documentar janelas de atributos, hiperparâmetros e métricas de teste.\n"
        "- Manter os artefatos históricos sincronizados com a base e as regras canônicas do projeto.\n",
    ])

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Markdown regenerado em {OUT_MD}")


if __name__ == "__main__":
    main()
