"""Interface Streamlit do TCC de previsão de interrupções elétricas."""

from __future__ import annotations

import sys
from pathlib import Path

INTERFACE_DIR = Path(__file__).resolve().parent
if str(INTERFACE_DIR) not in sys.path:
    sys.path.insert(0, str(INTERFACE_DIR))

import pandas as pd
import streamlit as st

from model_service import (
    DATASET_PATH,
    MODEL_OPTIONS,
    SCRIPT_CATALOG,
    SUPPORTED_HORIZONS,
    load_dataset,
    load_direct_results,
    load_reference_results,
    run_allowlisted_script,
    run_experiment,
    save_experiment,
)


st.set_page_config(
    page_title="Energia DF | Previsão de interrupções",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --energy-yellow: #f2b705;
        --energy-navy: #11253d;
        --energy-blue: #1d6f9f;
        --soft-bg: #f5f7fa;
    }
    .stApp { background: var(--soft-bg); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #10243b 0%, #173b58 100%);
    }
    [data-testid="stSidebar"] * { color: #f7fafc; }
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e1e8ef;
        border-left: 5px solid var(--energy-yellow);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        box-shadow: 0 4px 14px rgba(17, 37, 61, 0.06);
    }
    .hero {
        background: linear-gradient(120deg, #11253d 0%, #1d6f9f 100%);
        border-radius: 18px;
        color: white;
        padding: 1.4rem 1.7rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 26px rgba(17, 37, 61, 0.18);
    }
    .hero h1 { color: white; margin: 0 0 0.25rem 0; }
    .hero p { color: #e8f2f8; margin: 0; }
    .plain-card {
        background: white;
        border: 1px solid #e1e8ef;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0 1rem 0;
    }
    .small-note { color: #52677a; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_dataset() -> pd.DataFrame:
    return load_dataset(DATASET_PATH)


@st.cache_data(show_spinner=False)
def cached_reference_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_reference_results()


@st.cache_data(show_spinner=False)
def cached_direct_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_direct_results()


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pt_date(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%d/%m/%Y")


def metric_table(metrics: pd.DataFrame) -> None:
    display = metrics.rename(
        columns={
            "n": "Amostras",
            "R2": "R²",
        }
    ).copy()
    ordered = [
        column
        for column in (
            "Modelo",
            "Horizonte",
            "Mês",
            "Amostras",
            "MAE",
            "RMSE",
            "R²",
            "MAPE",
        )
        if column in display.columns
    ]
    st.dataframe(
        display[ordered],
        hide_index=True,
        width="stretch",
        column_config={
            "Horizonte": st.column_config.NumberColumn(
                "Horizonte (dias)",
                format="%d",
            ),
            "MAE": st.column_config.NumberColumn(format="%.2f"),
            "RMSE": st.column_config.NumberColumn(format="%.2f"),
            "R²": st.column_config.NumberColumn(format="%.3f"),
            "MAPE": st.column_config.NumberColumn("MAPE (%)", format="%.2f"),
        },
    )


def prediction_chart(predictions: pd.DataFrame, horizon: int, key: str) -> None:
    selected = predictions.loc[predictions["horizonte"].eq(horizon)].copy()
    selected["data_alvo"] = pd.to_datetime(selected["data_alvo"])
    actual = (
        selected[["data_alvo", "y_real"]]
        .drop_duplicates("data_alvo")
        .set_index("data_alvo")
        .rename(columns={"y_real": "Real"})
    )
    predicted = selected.pivot(
        index="data_alvo",
        columns="modelo",
        values="y_pred",
    )
    chart_data = actual.join(predicted, how="inner").sort_index()
    st.line_chart(
        chart_data,
        x_label="Data",
        y_label="Interrupções por dia",
        height=430,
    )



def monthly_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Agrega as métricas por mês, modelo e horizonte."""
    data = predictions.copy()
    data["data_alvo"] = pd.to_datetime(data["data_alvo"])
    data["Mês"] = data["data_alvo"].dt.to_period("M").dt.to_timestamp()
    rows = []
    for (model_name, horizon, month), group in data.groupby(
        ["modelo", "horizonte", "Mês"],
        sort=True,
    ):
        error = group["y_real"] - group["y_pred"]
        denominator = ((group["y_real"] - group["y_real"].mean()) ** 2).sum()
        rows.append(
            {
                "Modelo": model_name,
                "Horizonte": int(horizon),
                "Mês": month,
                "Amostras": len(group),
                "MAE": error.abs().mean(),
                "RMSE": (error.pow(2).mean()) ** 0.5,
                "R2": 1 - error.pow(2).sum() / denominator
                if denominator > 0
                else float("nan"),
                "MAPE": (
                    error.abs() / (group["y_real"].abs() + 1e-8)
                ).mean()
                * 100,
            }
        )
    return pd.DataFrame(rows)

def render_overview(df: pd.DataFrame) -> None:
    hero(
        "Previsão de interrupções elétricas",
        "Painel acessível para explorar os dados e comparar os modelos do TCC.",
    )
    st.markdown("### Visão geral dos dados")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dias analisados", f"{len(df):,}".replace(",", "."))
    col2.metric("Período", f"{df.index.min().year}–{df.index.max().year}")
    col3.metric("Média diária", f"{df['interrupcoes'].mean():.0f}")
    col4.metric("Variáveis preditoras", f"{df.shape[1] - 1}")

    default_start = max(df.index.min(), df.index.max() - pd.Timedelta(days=364))
    selected_period = st.date_input(
        "Período exibido no gráfico",
        value=(default_start.date(), df.index.max().date()),
        min_value=df.index.min().date(),
        max_value=df.index.max().date(),
        format="DD/MM/YYYY",
    )
    if len(selected_period) == 2:
        start, end = map(pd.Timestamp, selected_period)
        chart_data = df.loc[start:end, ["interrupcoes"]].rename(
            columns={"interrupcoes": "Interrupções"}
        )
        st.line_chart(
            chart_data,
            x_label="Data",
            y_label="Interrupções por dia",
            height=360,
        )

    st.markdown("### Resumo multi-horizonte")
    st.caption(
        "Comparação temporal dos horizontes de 1, 3, 7 e 14 dias em 352 datas-alvo."
    )
    try:
        metrics, _ = cached_reference_results()
    except (FileNotFoundError, ValueError) as error:
        st.warning(f"Os resultados de referência não puderam ser carregados: {error}")
        return

    best_rows = metrics.loc[metrics.groupby("Horizonte")["MAE"].idxmin()]
    best_text = " · ".join(
        f"{int(row.Horizonte)}d: {row.Modelo} (MAE {row.MAE:.2f})"
        for row in best_rows.itertuples()
    )
    st.info(f"Menor MAE em cada horizonte — {best_text}")

    comparison = metrics.pivot(
        index="Horizonte",
        columns="Modelo",
        values="MAE",
    ).sort_index()
    st.line_chart(
        comparison,
        x_label="Horizonte (dias)",
        y_label="MAE — menor é melhor",
        height=350,
    )
    metric_table(metrics)


def render_training(df: pd.DataFrame) -> None:
    hero(
        "Treinar e avaliar modelos",
        "Treine a previsão do próximo dia ou compare vários horizontes.",
    )
    st.markdown(
        """
        O **período de treinamento** determina quais datas-alvo ensinam o modelo.
        O **período de avaliação** fica separado e mede o desempenho em dados que
        o modelo não usou no ajuste.
        """
    )
    prediction_mode = st.radio(
        "Tipo de previsão",
        options=("Direta — próximo dia", "Multi-horizonte"),
        horizontal=True,
        help=(
            "Na previsão direta, o modelo usa informações do dia t para prever "
            "t+1. No modo multi-horizonte, é treinado um modelo independente "
            "para cada horizonte selecionado."
        ),
    )

    min_date = df.index.min().date()
    max_date = df.index.max().date()
    reference_test_start = pd.Timestamp(
        "2024-06-01"
        if prediction_mode == "Direta — próximo dia"
        else "2024-06-14"
    )
    if reference_test_start > df.index.max():
        reference_test_start = df.index.max() - pd.Timedelta(days=364)
    default_train_end = (reference_test_start - pd.Timedelta(days=1)).date()

    with st.form("training_form"):
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            training_period = st.date_input(
                "Período de treinamento",
                value=(min_date, default_train_end),
                min_value=min_date,
                max_value=max_date,
                format="DD/MM/YYYY",
            )
        with date_col2:
            evaluation_period = st.date_input(
                "Período de avaliação",
                value=(reference_test_start.date(), max_date),
                min_value=min_date,
                max_value=max_date,
                format="DD/MM/YYYY",
            )

        models = st.multiselect(
            "Modelos",
            options=list(MODEL_OPTIONS),
            default=["XGBoost"],
            help="ARIMAX está disponível como comparação experimental.",
        )
        if prediction_mode == "Direta — próximo dia":
            st.info("Previsão direta selecionada: dia t → dia t+1 (h = 1).")
            horizons = [1]
        else:
            horizons = st.multiselect(
                "Horizontes comparados",
                options=list(SUPPORTED_HORIZONS),
                default=list(SUPPORTED_HORIZONS),
                format_func=lambda value: (
                    f"{value} dia" if value == 1 else f"{value} dias"
                ),
            )

        with st.expander("Configurações das redes neurais"):
            st.caption(
                "Estas opções só afetam Bi-LSTM e Bi-GRU. "
                "A monografia usa 150 épocas e janela de 14 dias."
            )
            epochs = st.slider(
                "Épocas",
                min_value=5,
                max_value=150,
                value=30,
                step=5,
            )
            sequence_length = st.slider(
                "Tamanho da janela histórica (dias)",
                min_value=7,
                max_value=30,
                value=14,
            )

        submitted = st.form_submit_button(
            "Treinar previsão direta"
            if prediction_mode == "Direta — próximo dia"
            else "Treinar multi-horizonte",
            type="primary",
            width="stretch",
        )

    if "ARIMAX" in models:
        st.warning(
            "ARIMAX experimental: nesta avaliação, as covariáveis observadas na "
            "data de origem são fornecidas ao modelo. Uso operacional exigiria "
            "previsões meteorológicas compatíveis."
        )
    if any(model in {"Bi-LSTM", "Bi-GRU"} for model in models):
        st.caption(
            "Redes neurais podem levar vários minutos. Para reproduzir a "
            "configuração científica do trabalho, selecione 150 épocas."
        )

    if submitted:
        if len(training_period) != 2 or len(evaluation_period) != 2:
            st.error("Selecione o início e o fim dos dois períodos.")
        else:
            train_start, train_end = training_period
            test_start, test_end = evaluation_period
            messages: list[str] = []
            progress_box = st.empty()

            def update_progress(message: str) -> None:
                messages.append(message)
                progress_box.code("\n".join(messages[-8:]), language=None)

            config = {
                "tipo_previsao": prediction_mode,
                "modelos": models,
                "horizontes": horizons,
                "treino_inicio": train_start,
                "treino_fim": train_end,
                "avaliacao_inicio": test_start,
                "avaliacao_fim": test_end,
                "epocas": epochs,
                "janela_dias": sequence_length,
            }
            try:
                with st.spinner("Treinando e calculando as métricas..."):
                    predictions, metrics = run_experiment(
                        df,
                        models=models,
                        horizons=horizons,
                        train_start=train_start,
                        train_end=train_end,
                        test_start=test_start,
                        test_end=test_end,
                        epochs=epochs,
                        sequence_length=sequence_length,
                        progress=update_progress,
                    )
                    output_dir = save_experiment(predictions, metrics, config)
                st.session_state["last_experiment"] = {
                    "predictions": predictions,
                    "metrics": metrics,
                    "output_dir": str(output_dir),
                    "prediction_mode": prediction_mode,
                }
                st.success("Treinamento concluído e resultados salvos.")
            except Exception as error:
                st.error(f"Não foi possível concluir o treinamento: {error}")

    result = st.session_state.get("last_experiment")
    if result:
        st.divider()
        result_mode = result.get("prediction_mode", "Execução anterior")
        st.markdown(f"### Resultado da última execução — {result_mode}")
        metric_table(result["metrics"])
        available_horizons = sorted(result["predictions"]["horizonte"].unique())
        if len(available_horizons) == 1:
            selected_horizon = available_horizons[0]
            st.caption(f"Horizonte avaliado: {selected_horizon} dia.")
        else:
            selected_horizon = st.selectbox(
                "Horizonte exibido",
                options=available_horizons,
                format_func=lambda value: (
                    f"{value} dia" if value == 1 else f"{value} dias"
                ),
                key="training_result_horizon",
            )
        prediction_chart(
            result["predictions"],
            selected_horizon,
            key="training_prediction_chart",
        )
        st.markdown("#### Desempenho por mês")
        monthly = monthly_metric_table(result["predictions"])
        monthly = monthly.loc[monthly["Horizonte"].eq(selected_horizon)]
        monthly_chart = monthly.pivot(
            index="Mês",
            columns="Modelo",
            values="MAE",
        )
        st.line_chart(
            monthly_chart,
            x_label="Mês",
            y_label="MAE mensal — menor é melhor",
            height=300,
        )
        st.caption(f"Arquivos salvos em: {result['output_dir']}")

        download_col1, download_col2 = st.columns(2)
        download_col1.download_button(
            "Baixar previsões (CSV)",
            data=result["predictions"].to_csv(index=False).encode("utf-8"),
            file_name="previsoes.csv",
            mime="text/csv",
            width="stretch",
        )
        download_col2.download_button(
            "Baixar métricas (CSV)",
            data=result["metrics"].to_csv(index=False).encode("utf-8"),
            file_name="metricas.csv",
            mime="text/csv",
            width="stretch",
        )


def render_direct_prediction() -> None:
    hero(
        "Previsão direta — próximo dia",
        "Resultados da tarefa dia t → dia t+1, separados da análise multi-horizonte.",
    )
    st.markdown(
        """
        Nesta tarefa, cada modelo recebe as informações disponíveis no **dia t**
        e estima o número de interrupções no **dia seguinte (t+1)**. Estes
        resultados vêm dos arquivos gerados pelos scripts principais de XGBoost,
        Bi-LSTM e Bi-GRU, e não da tabela multi-horizonte.
        """
    )
    try:
        metrics, predictions = cached_direct_results()
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Resultados diretos não encontrados: {error}")
        return

    best = metrics.loc[metrics["MAE"].idxmin()]
    col1, col2, col3 = st.columns(3)
    col1.metric("Melhor MAE", best["Modelo"])
    col2.metric("MAE", f"{best['MAE']:.2f}")
    col3.metric("Dias avaliados", f"{int(best['n'])}")

    st.markdown("### Métricas da previsão direta")
    metric_table(metrics)
    st.markdown("### Real versus previsto no dia seguinte")
    prediction_chart(predictions, 1, key="direct_prediction_chart")

    st.markdown("### Desempenho mês a mês")
    monthly = monthly_metric_table(predictions)
    monthly_chart = monthly.pivot(
        index="Mês",
        columns="Modelo",
        values="MAE",
    )
    st.line_chart(
        monthly_chart,
        x_label="Mês",
        y_label="MAE mensal — menor é melhor",
        height=340,
    )

def render_comparison() -> None:
    hero(
        "Comparação multi-horizonte",
        "Veja como cada modelo se comporta ao prever de 1 a 14 dias à frente.",
    )
    try:
        metrics, predictions = cached_reference_results()
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Resultados não encontrados: {error}")
        return

    metric_choice = st.radio(
        "Métrica",
        options=["MAE", "RMSE", "MAPE", "R2"],
        horizontal=True,
        format_func=lambda value: "R²" if value == "R2" else value,
    )
    comparison = metrics.pivot(
        index="Horizonte",
        columns="Modelo",
        values=metric_choice,
    ).sort_index()
    st.line_chart(
        comparison,
        x_label="Horizonte (dias)",
        y_label=f"{'R²' if metric_choice == 'R2' else metric_choice}",
        height=390,
    )
    metric_table(metrics)

    st.markdown("### Real versus previsto")
    horizon = st.select_slider(
        "Horizonte",
        options=list(SUPPORTED_HORIZONS),
        value=1,
        format_func=lambda value: f"{value} dia" if value == 1 else f"{value} dias",
    )
    prediction_chart(predictions, horizon, key="reference_prediction_chart")
    st.caption(
        "O valor real é igual para todos os modelos na mesma data-alvo; "
        "as linhas coloridas mostram as previsões."
    )

    st.markdown("### Comparação mês a mês")
    st.caption(
        "Esta visão ajuda a identificar sazonalidade no erro e períodos em que "
        "cada modelo funciona melhor ou pior."
    )
    monthly_choice = st.selectbox(
        "Métrica mensal",
        options=["MAE", "RMSE", "MAPE", "R2"],
        format_func=lambda value: "R²" if value == "R2" else value,
    )
    monthly = monthly_metric_table(predictions)
    monthly = monthly.loc[monthly["Horizonte"].eq(horizon)]
    monthly_chart = monthly.pivot(
        index="Mês",
        columns="Modelo",
        values=monthly_choice,
    )
    st.line_chart(
        monthly_chart,
        x_label="Mês",
        y_label="R²" if monthly_choice == "R2" else monthly_choice,
        height=340,
    )
    monthly_display = monthly.copy()
    monthly_display["Mês"] = monthly_display["Mês"].dt.strftime("%m/%Y")
    metric_table(monthly_display)


def render_scripts() -> None:
    hero(
        "Executar análises",
        "Reproduza tarefas do projeto sem usar o terminal.",
    )
    st.warning(
        "Alguns scripts atualizam arquivos em Fonte/results. Não feche esta "
        "página enquanto uma execução estiver em andamento."
    )
    keys = list(SCRIPT_CATALOG)
    selected_key = st.selectbox(
        "Tarefa",
        options=keys,
        format_func=lambda key: SCRIPT_CATALOG[key]["label"],
    )
    selected = SCRIPT_CATALOG[selected_key]
    st.markdown(
        f"""
        <div class="plain-card">
          <strong>{selected["label"]}</strong><br>
          {selected["description"]}<br>
          <span class="small-note">Tempo estimado: {selected["duration"]}.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Executar tarefa", type="primary"):
        try:
            with st.spinner(f"Executando: {selected['label']}..."):
                report = run_allowlisted_script(selected_key)
            st.success("Tarefa concluída.")
            st.code(report, language=None)
            cached_dataset.clear()
            cached_reference_results.clear()
            cached_direct_results.clear()
        except Exception as error:
            st.error(str(error))


def render_about(df: pd.DataFrame) -> None:
    hero(
        "Sobre a ferramenta",
        "Um guia curto para interpretar a interface corretamente.",
    )
    st.markdown(
        f"""
        ### O que esta versão faz

        - explora o dataset processado de **{pt_date(df.index.min())}** a
          **{pt_date(df.index.max())}**;
        - treina XGBoost, Bi-LSTM, Bi-GRU e um ARIMAX experimental;
        - permite separar faixas de treinamento e avaliação;
        - apresenta separadamente a previsão direta do próximo dia;
        - compara modelos independentes nos horizontes de 1, 3, 7 e 14 dias;
        - executa scripts cadastrados e exibe suas mensagens;
        - salva cada experimento em `Fonte/results/interface`.

        ### Como ler as métricas

        - **MAE**: erro absoluto médio em interrupções por dia; menor é melhor.
        - **RMSE**: penaliza mais fortemente erros grandes; menor é melhor.
        - **R²**: proporção da variação explicada; mais próximo de 1 é melhor.
        - **MAPE**: erro percentual médio; menor é melhor.

        ### Limites importantes

        A ferramenta usa o dataset já processado no repositório. Ela não baixa
        automaticamente novos dados da ANEEL ou do INMET. Os resultados da tela
        são avaliações históricas e não devem ser tratados, sem validação
        adicional, como alertas operacionais em tempo real.
        """
    )


df = cached_dataset()

with st.sidebar:
    st.markdown("## ⚡ Energia DF")
    st.caption("Interface do TCC")
    page = st.radio(
        "Navegação",
        options=(
            "Visão geral",
            "Previsão direta",
            "Treinar modelos",
            "Multi-horizonte",
            "Executar análises",
            "Sobre",
        ),
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(
        f"Dados: {pt_date(df.index.min())} — {pt_date(df.index.max())}"
    )

if page == "Visão geral":
    render_overview(df)
elif page == "Previsão direta":
    render_direct_prediction()
elif page == "Treinar modelos":
    render_training(df)
elif page == "Multi-horizonte":
    render_comparison()
elif page == "Executar análises":
    render_scripts()
else:
    render_about(df)
