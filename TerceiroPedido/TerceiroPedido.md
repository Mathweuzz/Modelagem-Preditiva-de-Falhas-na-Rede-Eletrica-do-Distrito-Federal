# Terceira Entrega — Ajustes Visuais, Correlações e Vento (INMET)

## Principais ajustes solicitados e implementados

- **Médias móveis**: gráficos diários passaram a ser gerados em **janelas de 1 ano** (melhor legibilidade).
- **Padronização de cores**: interrupções em **vermelho**, temperatura em **azul** e precipitação em **azul forte**.
- **Interrupções x temperatura (semanal)**: geração por **intervalos anuais**.
- **Scatters mensais**: cada ponto representa **1 mês**, com **cor por ano/gradiente temporal** e **linha de regressão + R²**.
- **Previsão (baselines)**: visualização do período de teste em **janelas de 1 ano**.
- **Consumo**: visualização também em **GWh** para evitar notação científica no eixo e facilitar interpretação.
- **Vento (INMET)**: criação de variáveis diárias (velocidade, rajada e direção), integração com interrupções e análise em escalas diária/semanal/mensal.

## Bases utilizadas nesta entrega

- `dados/base_diaria_interrupcoes_clima.csv` — período: **2017-01-01 a 2025-05-31**; linhas: **3073**
- `dados/base_diaria_interrupcoes_clima_vento.csv` — período: **2017-01-01 a 2025-05-31**; linhas: **3073**
- `dados/base_mensal_interrupcoes_clima_consumo.csv` — período: **2017-01-01 a 2025-05-01**; meses: **101**
- `dados/previsoes_diarias_baselines.csv` — período: **2017-01-01 a 2025-05-31**; linhas: **3073**

## Artefatos gerados por tarefa

### T1 — Médias móveis diárias por ano (1 ano por gráfico)

**Pasta:** `graficos/T1_mm_1ano/`

- `mm_diario_interrupcoes_2017.png`
- `mm_diario_interrupcoes_2018.png`
- `mm_diario_interrupcoes_2019.png`
- `mm_diario_interrupcoes_2020.png`
- `mm_diario_interrupcoes_2021.png`
- `mm_diario_interrupcoes_2022.png`
- `mm_diario_interrupcoes_2023.png`
- `mm_diario_interrupcoes_2024.png`
- `mm_diario_interrupcoes_2025.png`
- `mm_diario_precipitacao_2017.png`
- `mm_diario_precipitacao_2018.png`
- `mm_diario_precipitacao_2019.png`
- `mm_diario_precipitacao_2020.png`
- `mm_diario_precipitacao_2021.png`
- `mm_diario_precipitacao_2022.png`
- `mm_diario_precipitacao_2023.png`
- `mm_diario_precipitacao_2024.png`
- `mm_diario_precipitacao_2025.png`
- `mm_diario_temperatura_2017.png`
- `mm_diario_temperatura_2018.png`
- `mm_diario_temperatura_2019.png`
- `mm_diario_temperatura_2020.png`
- `mm_diario_temperatura_2021.png`
- `mm_diario_temperatura_2022.png`
- `mm_diario_temperatura_2023.png`
- `mm_diario_temperatura_2024.png`
- `mm_diario_temperatura_2025.png`

### T3 — Interrupções x Temperatura (semanal) por ano

**Pasta:** `graficos/T3_semanal_temp_ano/`

- `semanal_interrupcoes_vs_temperatura_2017.png`
- `semanal_interrupcoes_vs_temperatura_2018.png`
- `semanal_interrupcoes_vs_temperatura_2019.png`
- `semanal_interrupcoes_vs_temperatura_2020.png`
- `semanal_interrupcoes_vs_temperatura_2021.png`
- `semanal_interrupcoes_vs_temperatura_2022.png`
- `semanal_interrupcoes_vs_temperatura_2023.png`
- `semanal_interrupcoes_vs_temperatura_2024.png`
- `semanal_interrupcoes_vs_temperatura_2025.png`

### T4 — Interrupções x Precipitação (semanal e mensal) com cores padronizadas

**Pasta:** `graficos/T4_precipitacao/`

- `interrupcoes_vs_precipitacao_mensal.png`
- `interrupcoes_vs_precipitacao_semanal.png`

### T5 — Scatters mensais com cor por ano/gradiente + regressão

**Pasta:** `graficos/T5_scatter_regressao/`

- `scatter_consumo_vs_interrupcoes_gradiente_tempo.png`
- `scatter_consumo_vs_interrupcoes_por_ano.png`
- `scatter_temperatura_vs_consumo_gradiente_tempo.png`
- `scatter_temperatura_vs_consumo_por_ano.png`

### T6 — Previsão (baselines) no teste com zoom de 1 ano

**Pasta:** `graficos/T6_previsao_zoom_1ano/`

- `previsao_baselines_teste_2023.png`
- `previsao_baselines_teste_2024.png`
- `previsao_baselines_teste_2025.png`

### T8 — Vento diário (INMET) integrado e gráficos diários

**Pasta:** `graficos/T8_vento/`

- `diario_interrupcoes_vs_rajada_max.png`
- `diario_interrupcoes_vs_vento_vel_media.png`

### T9 — Vento agregado semanal/mensal e gráficos

**Pasta:** `graficos/T9_vento_agregados/`

- `mensal_interrupcoes_vs_direcao_media.png`
- `mensal_interrupcoes_vs_rajada_max.png`
- `mensal_interrupcoes_vs_vento_vel_media.png`
- `semanal_interrupcoes_vs_direcao_media.png`
- `semanal_interrupcoes_vs_rajada_max.png`
- `semanal_interrupcoes_vs_vento_vel_media.png`

## Correlações (Pearson) — resumo consolidado

### Destaques (maiores correlações em módulo)

| nivel_temporal | variavel | pearson_r |
| --- | --- | ---: |
| mensal | vento_dir_cos | 0.569979 |
| mensal | vento_dir_sin | -0.522178 |
| mensal | precipitacao_total_mm | 0.538558 |
| semanal | precipitacao_total_mm | 0.495241 |
| semanal | vento_dir_sin | -0.436665 |
| mensal | consumo_total_kwh | 0.476395 |
| semanal | vento_dir_cos | 0.394814 |
| diario | vento_dir_sin | -0.345359 |
| semanal | vento_rajada_max_ms | 0.401391 |
| diario | precipitacao_total_mm | 0.347621 |
| mensal | vento_velocidade_media_ms | -0.310225 |
| mensal | vento_rajada_max_ms | 0.448605 |

### Tabela completa

| nivel_temporal | variavel | pearson_r |
| --- | --- | ---: |
| diario | temperatura_media | 0.101885 |
| diario | precipitacao_total_mm | 0.347621 |
| diario | vento_velocidade_media_ms | -0.149055 |
| diario | vento_velocidade_max_ms | 0.033575 |
| diario | vento_rajada_max_ms | 0.254401 |
| diario | vento_dir_sin | -0.345359 |
| diario | vento_dir_cos | 0.210274 |
| semanal | temperatura_media | 0.250893 |
| semanal | precipitacao_total_mm | 0.495241 |
| semanal | vento_velocidade_media_ms | -0.249761 |
| semanal | vento_velocidade_max_ms | 0.072363 |
| semanal | vento_rajada_max_ms | 0.401391 |
| semanal | vento_dir_sin | -0.436665 |
| semanal | vento_dir_cos | 0.394814 |
| mensal | temperatura_media | 0.393729 |
| mensal | precipitacao_total_mm | 0.538558 |
| mensal | vento_velocidade_media_ms | -0.310225 |
| mensal | vento_velocidade_max_ms | 0.069546 |
| mensal | vento_rajada_max_ms | 0.448605 |
| mensal | vento_dir_sin | -0.522178 |
| mensal | vento_dir_cos | 0.569979 |
| mensal | consumo_total_kwh | 0.476395 |

## Interpretação resumida dos achados

- **Agregação canônica**: nem todas as associações aumentam monotonicamente entre escalas.
- **Precipitação**: $r$ passa de 0,348 no diário para 0,495 no semanal e 0,539 no mensal.
- **Consumo**: a associação mensal com o alvo é 0,476; o consumo não integra as entradas dos modelos.
- **Direção do vento**: componentes seno/cosseno têm $r=-0,522$ e $r=0,570$ em escala mensal.
- **Interpretação**: os coeficientes descrevem associação e não demonstram causalidade.


## Modelos de previsão (Deep Learning) — LSTM e GRU (PyTorch)

Nesta etapa foram treinados modelos **LSTM** e **GRU** usando divisão temporal **sem vazamento**: para prever o dia *t*, o modelo recebe apenas informações históricas até *t-1* (janela lookback) e variáveis meteorológicas/vento como covariáveis.

### Métricas (treino e teste)

| modelo   | conjunto   |     MAE |     RMSE |       R2 |
|:---------|:-----------|--------:|---------:|---------:|
| LSTM     | treino     | 47.9353 |  78.6596 | 0.477103 |
| LSTM     | teste      | 61.8356 | 100.663  | 0.376483 |
| GRU      | treino     | 48.2667 |  78.0922 | 0.48462  |
| GRU      | teste      | 61.7902 |  98.064  | 0.408262 |

### Artefatos

- `dados/previsoes_dl_lstm_gru.csv`
- `dados/metricas_dl_lstm_gru.csv`
- `graficos/T12_modelos_dl/previsao_dl_zoom_1ano.png`
- `graficos/T13_comparacao/comparacao_previsoes_zoom_1ano.png`


**Resumo:** no conjunto de teste, os modelos LSTM/GRU superaram as baselines (persistência e MM7), com destaque para a **GRU**.
