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

| nivel_temporal   | variavel_x        | variavel_y                |   pearson_r |
|:-----------------|:------------------|:--------------------------|------------:|
| mensal           | interrupcoes      | vento_direcao_media_gr    |    0.58791  |
| mensal           | consumo_total_kwh | temperatura_media         |    0.555031 |
| mensal           | interrupcoes      | precipitacao_total_mm     |    0.538558 |
| semanal          | interrupcoes      | vento_direcao_media_gr    |    0.496223 |
| semanal          | interrupcoes      | precipitacao_total_mm     |    0.476843 |
| mensal           | interrupcoes      | consumo_total_kwh         |    0.476395 |
| mensal           | interrupcoes      | temperatura_media         |    0.393729 |
| mensal           | interrupcoes      | vento_rajada_max_ms       |    0.37889  |
| diario           | interrupcoes      | vento_direcao_media_gr    |    0.366492 |
| mensal           | interrupcoes      | vento_velocidade_media_ms |   -0.363507 |
| diario           | interrupcoes      | precipitacao_total_mm     |    0.347405 |
| semanal          | interrupcoes      | vento_rajada_max_ms       |    0.340133 |

### Tabela completa

| nivel_temporal   | variavel_x        | variavel_y                |   pearson_r |
|:-----------------|:------------------|:--------------------------|------------:|
| diario           | interrupcoes      | vento_direcao_media_gr    |   0.366492  |
| diario           | interrupcoes      | precipitacao_total_mm     |   0.347405  |
| diario           | interrupcoes      | vento_rajada_max_ms       |   0.253027  |
| diario           | interrupcoes      | vento_velocidade_media_ms |  -0.193947  |
| diario           | interrupcoes      | vento_direcao_moda_gr     |   0.181191  |
| diario           | interrupcoes      | temperatura_media         |   0.101885  |
| diario           | interrupcoes      | vento_velocidade_max_ms   |   0.0157279 |
| mensal           | interrupcoes      | vento_direcao_media_gr    |   0.58791   |
| mensal           | consumo_total_kwh | temperatura_media         |   0.555031  |
| mensal           | interrupcoes      | precipitacao_total_mm     |   0.538558  |
| mensal           | interrupcoes      | consumo_total_kwh         |   0.476395  |
| mensal           | interrupcoes      | temperatura_media         |   0.393729  |
| mensal           | interrupcoes      | vento_rajada_max_ms       |   0.37889   |
| mensal           | interrupcoes      | vento_velocidade_media_ms |  -0.363507  |
| mensal           | interrupcoes      | vento_direcao_moda_gr     |  -0.185887  |
| mensal           | interrupcoes      | vento_velocidade_max_ms   |  -0.0835723 |
| semanal          | interrupcoes      | vento_direcao_media_gr    |   0.496223  |
| semanal          | interrupcoes      | precipitacao_total_mm     |   0.476843  |
| semanal          | interrupcoes      | vento_rajada_max_ms       |   0.340133  |
| semanal          | interrupcoes      | vento_velocidade_media_ms |  -0.313593  |
| semanal          | interrupcoes      | vento_direcao_moda_gr     |  -0.0836208 |
| semanal          | interrupcoes      | vento_velocidade_max_ms   |  -0.0692043 |

## Interpretação resumida dos achados

- **Agregação aumenta clareza do padrão**: a relação entre interrupções e variáveis meteorológicas tende a ficar mais forte em escalas **semanal/mensal** do que no diário.
- **Precipitação**: correlação cresce de **diário (~0,35)** para **semanal (~0,48)** e **mensal (~0,54)**.
- **Consumo e temperatura (mensal)**: correlação moderada/alta (**~0,56**), e interrupções também se correlacionam com consumo (**~0,48**).
- **Vento**: após limpeza de valores inválidos do INMET, a **direção média do vento** apresenta correlação relevante com interrupções, especialmente em escala **mensal (~0,59)**; semanal também é significativa (**~0,50**).
- **Observação metodológica**: direção do vento é uma variável circular (0–360°); a média simples é uma aproximação inicial e pode ser refinada com estatística circular, se necessário.


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
