# Roteiro do Código-Fonte — Para o Prof. Jan Mendonça Corrêa

**TCC:** Predição de Interrupções no Fornecimento de Energia Elétrica no DF
**Autores:** Giovanni Minari Zanetti, Mateus Gomes de Araújo
**Data:** abril/2026

Este documento mapeia **cada figura, tabela e métrica da monografia ao script Python que a gerou e ao arquivo de dados que foi consumido**. O objetivo é permitir ao orientador reproduzir qualquer artefato do trabalho.

> **Sobre a Figura 2.1** (diagrama da célula LSTM, Cap. 2): foi construída por nós em **TikZ puro** dentro do próprio arquivo `Monografia/tex/2_FundamentacaoTeorica.tex` (linhas 204-254). Não é uma imagem externa — é renderizada pelo `pdflatex` durante a compilação.

---

## 1. Como reproduzir

A partir da raiz do repositório, com Python ≥ 3.10 e as dependências instaladas, a reprodução oficial deve usar o orquestrador, que valida todos os artefatos antes de promovê-los:

```bash
Fonte/venv/bin/python Fonte/run_pipeline.py \
  --interruptions /caminho/dados_completos_brasilia.csv \
  --inmet-dir /caminho/dados_clima-inmet_limpos
```

Os detalhes dos argumentos, etapas e artefatos obrigatórios estão em `Fonte/README.md`.

Tempo total estimado: ~15-25 min em CPU (Intel i5-9300HF).

---

## 2. Mapeamento Figura → Script → Dados

### Capítulo 2 — Fundamentação Teórica

| Figura | Descrição | Origem |
|---|---|---|
| **Fig. 2.1** | Diagrama célula LSTM (portões F/I/O, Cell State) | **TikZ** em `Monografia/tex/2_FundamentacaoTeorica.tex` (linhas 204-254) |
| **Fig. 2.2** | Comparativo estrutural LSTM vs GRU | **TikZ** em `Monografia/tex/2_FundamentacaoTeorica.tex` (linhas 269+) |

### Capítulo 3 — Metodologia

Diagramas de fluxo (pipeline de dados, arquitetura experimental) — todos **TikZ** em `Monografia/tex/3_Metodologia.tex`.

### Capítulo 4 — Resultados

| Figura | Descrição | Script | Entrada |
|---|---|---|---|
| `serie_temporal_completa` | Série diária + SMA-30 + corte treino/teste | `Fonte/src/04_eda_basica.py::plot_serie_temporal_completa` | `base_diaria_interrupcoes_clima_vento.csv` |
| `distribuicao_interrupcoes` | Histograma + KDE (média/mediana) | `Fonte/src/04_eda_basica.py::plot_distribuicao_interrupcoes` | idem |
| `evolucao_anual_interrupcoes` | Média ± std por ano (treino azul / teste vermelho) | `Fonte/src/04_eda_basica.py::plot_evolucao_anual` | idem |
| `decomposicao_interrupcoes` | Decomposição STL aditiva (tendência/sazonal/resíduo, período=365) | `Fonte/src/01_eda_sazonalidade.py::plot_decomposition` | idem |
| `autocorrelacao_interrupcoes` | ACF + PACF (até lag 60) | `Fonte/src/01_eda_sazonalidade.py::plot_autocorrelation` | idem |
| `eda_heatmap_pearson` | Matriz Pearson clima × interrupções | `Fonte/src/models/script_exploration_pipeline.py` | idem |
| `correlacao_spearman` | Matriz Spearman não-linear | `Fonte/src/02_correlacoes_nao_lineares.py::plot_correlation_matrix(method='spearman')` | idem |
| `correlacoes_escala_temporal` | Pearson em 3 escalas (diária/semanal/mensal) | `Fonte/src/05_correlacoes_unificadas.py` | idem |
| `cross_corr_chuva_interrupcoes` | Cross-correlation chuva(t-lag) × interrupções | `Fonte/src/02_correlacoes_nao_lineares.py::plot_cross_correlation` | idem |
| `cross_corr_vento_interrupcoes` | Cross-correlation rajada(t-lag) × interrupções | idem | idem |
| `eda_scatter_ventos` | Scatter rajada × interrupções (cor=chuva) | `Fonte/src/models/script_exploration_pipeline.py` | idem |
| `eda_boxplot_sazonalidade` | Boxplot mensal | idem | idem |
| `eda_violin_anomalias` | Violin temperatura × faixas descritivas do volume diário | `Fonte/src/04_eda_basica.py::plot_violin_anomalias` | idem |
| `interrupcoes_vs_precipitacao_mensal` | Linha + barra precipitação mensal | `TerceiroPedido/scripts/t4_precipitacao_cores.py` | `aggregados_mensal_interrupcoes_precipitacao.csv` |
| `interrupcoes_vs_precipitacao_semanal` | Linha + barra precipitação semanal | idem | `aggregados_semanal_interrupcoes_precipitacao.csv` |
| `mensal_interrupcoes_vs_rajada_max` | Linha + barra rajada mensal | `TerceiroPedido/scripts/t9_vento_agregado_semana_mes.py` | `aggregados_mensal_interrupcoes_vento.csv` |
| `diario_interrupcoes_vs_rajada_max` | Série diária integrada vento × interrupções | `TerceiroPedido/scripts/t8_vento_diario_integrado.py` | `Fonte/data/base_diaria_interrupcoes_clima_vento.csv` |
| `mm_diario_interrupcoes_2021` | Média móvel diária 2021 | `TerceiroPedido/scripts/t1_mm_1ano_cores.py` | `base_diaria_interrupcoes_clima_mm.csv` |
| `mm_diario_interrupcoes_2023` | Média móvel diária 2023 | idem | idem |
| `mm_diario_temperatura_2023` | Temperatura suavizada 2023 | idem | idem |
| `mm_diario_precipitacao_2023` | Precipitação suavizada 2023 | idem | idem |
| `scatter_consumo_vs_interrupcoes_por_ano` | Scatter consumo×interrupções com regressão | `TerceiroPedido/scripts/t5_scatter_cor_por_ano_regressao.py` | `base_mensal_interrupcoes_clima_consumo.csv` |
| `scatter_temperatura_vs_consumo_por_ano` | Scatter temperatura×consumo com regressão | idem | idem |
| `feature_importance_xgboost` | Top-15 features por gain (XGBoost) | `Fonte/src/models/baseline_xgboost.py::evaluate_and_plot` | `dataset_engenharia_features.csv` |
| `learning_curve_lstm_bidirecional` | Curva de aprendizado MSE (Bi-LSTM) | `Fonte/src/models/lstm_bidirecional.py::plot_loss` | idem |
| `learning_curve_gru_bidirecional` | Curva de aprendizado MSE (Bi-GRU) | `Fonte/src/models/gru_avancada.py` (via `plot_loss`) | idem |
| `ts_pred_xgboost` | Real vs previsto (série, XGBoost) | `Fonte/src/models/baseline_xgboost.py::evaluate_and_plot` | idem |
| `ts_pred_lstm_bi` | Real vs previsto (série, Bi-LSTM) | `Fonte/src/models/lstm_bidirecional.py` (via `evaluate_and_plot`) | idem |
| `ts_pred_gru_bi` | Real vs previsto (série, Bi-GRU) | `Fonte/src/models/gru_avancada.py` (via `evaluate_and_plot`) | idem |
| `scatter_pred_xgboost` | Real vs previsto (dispersão, XGBoost) | `evaluate_and_plot` | idem |
| `scatter_pred_lstm_bi` | Dispersão Bi-LSTM | idem | idem |
| `scatter_pred_gru_bi` | Dispersão Bi-GRU | idem | idem |
| `kde_residuos_modelos` | KDE dos resíduos dos 3 modelos | `Fonte/src/models/advanced_plots.py::plot_residual_kde` | `predictions_*.csv` (saída dos modelos) |
| `scatter_heteroscedasticity` | |Erro| × volume real (Bi-LSTM) | `Fonte/src/models/advanced_plots.py::plot_heteroscedasticity_scatter` | idem |

---

## 3. Mapeamento Tabela → Origem dos números

| Tabela | Cap. | Origem |
|---|---|---|
| Estatísticas descritivas das interrupções | 4 | Calculada via `pandas.DataFrame.describe()` sobre `base_diaria_interrupcoes_clima_vento.csv` |
| Métricas dos modelos (MAE, RMSE, R², MAPE) | 4 | `results/ml/metrics_xgboost.csv`, `metrics_lstm_bi.csv`, `metrics_gru_bi.csv` (gerados por `evaluate_and_plot`) |
| Trabalhos relacionados (literatura) | 1, 5 | Compilada manualmente a partir das referências `.bib` |
| Correlações consolidadas | 4 | `Fonte/data/correlacoes_consolidadas.csv` (saída de `Fonte/src/05_correlacoes_unificadas.py`; o script `TerceiroPedido/scripts/t10_correlacoes_consolidadas.py` apenas mantém a cópia histórica sincronizada) |

---

## 4. Bases de dados (todas em `Fonte/data/`)

| Arquivo | Conteúdo | Período |
|---|---|---|
| `base_diaria_interrupcoes_clima.csv` | Diária: data, interrupcoes, temperatura, precipitação | 2017-01-01 a 2025-05-31 |
| `base_diaria_interrupcoes_clima_vento.csv` | Diária + vento (velocidade média/máxima, rajada e direção em seno/cosseno) | idem |
| `base_diaria_interrupcoes_clima_mm.csv` | Diária + médias móveis 7d/14d (interrupções/temp/chuva) | idem |
| `base_mensal_interrupcoes_clima_consumo.csv` | Mensal: + consumo total kWh (SAMP) | idem |
| `dataset_engenharia_features.csv` | **Dataset final dos modelos**: 40 features + alvo | 2017-01-08 a 2025-05-31 (3.066 dias) |
| `previsoes_diarias_baselines.csv` | Previsões dos baselines — **legado** (ver `Fonte/data/legado/`) | — |
| `previsoes_dl_lstm_gru.csv` | Previsões LSTM/GRU de modelos antigos — **legado** (ver `Fonte/data/legado/`) | — |
| `vento_diario_brasilia.csv` | Estatísticas diárias de vento (INMET A001) | 2017-2025 |
| `aggregados_*.csv` / `correlacoes_*.csv` | Agregações temporais e tabelas auxiliares | — |

### Origem dos dados brutos (não incluídos)

- **INMET** — https://portal.inmet.gov.br/dadoshistoricos (estação automática A001 / Brasília)
- **ANEEL** — Relatórios PRODIST, indicadores de continuidade da Neoenergia Brasília
- **CCEE/SAMP** — consumo mensal por classe consumidora no DF

---

## 5. Reprodutibilidade

- **XGBoost**: `random_state=42`. Resultado **determinístico**. Protocolo h=1 direto (separação pela data-alvo): MAE=61.47 / RMSE=99.73 / R²=0.409 / MAPE=20.38% — conforme reportado no Cap. 4.
- **PyTorch (LSTM/GRU)**: pequenas variações entre execuções são esperadas (non-determinism do CUDA/cuDNN, ordem dos batches no DataLoader). As métricas reportadas no Cap. 4 foram obtidas no ambiente descrito no Apêndice (Reprodutibilidade) — versão dos pacotes em `Apendices`.

## 6. Limitações e ressalvas

Os scripts em `SegundoPedido/` e `TerceiroPedido/` foram criados como entregas iterativas durante o desenvolvimento (atendendo a três reuniões com o orientador). Eles foram preservados na estrutura original para manter rastreabilidade histórica do processo. Os mais relevantes para a monografia estão listados na seção 2.

Nos scripts da segunda entrega que dependem de arquivos brutos não versionados, a raiz externa pode ser definida com `TCC_RAW_DATA_ROOT=/caminho/dos/dados`. Sem essa variável, a raiz do próprio repositório é usada como padrão. Os scripts localizam o projeto por marcadores de diretório, sem caminhos absolutos pessoais nem contagem fixa de níveis.
