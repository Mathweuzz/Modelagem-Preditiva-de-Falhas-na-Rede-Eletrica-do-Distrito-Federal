# Terceira Entrega — Ajustes Visuais, Correlações e Vento (INMET)

Este documento resume as alterações e os resultados produzidos na terceira entrega.

## Principais ajustes implementados

- **Médias móveis**: gráficos diários em janelas de um ano, para melhorar a legibilidade.
- **Padronização de cores**: interrupções em vermelho, temperatura em azul e precipitação em azul-escuro.
- **Agregações**: visualizações semanais e mensais, além de dispersões mensais com regressão e $R^2$.
- **Consumo**: visualização em GWh para facilitar a leitura dos eixos.
- **Vento**: variáveis diárias da estação A001, com direção representada por seno e cosseno e agregada circularmente.

## Bases utilizadas nesta entrega

- `dados/base_diaria_interrupcoes_clima.csv` — período: **2017-01-01 a 2025-05-31**; linhas: **3073**
- `dados/base_diaria_interrupcoes_clima_vento.csv` — período: **2017-01-01 a 2025-05-31**; linhas: **3073**
- `dados/base_mensal_interrupcoes_clima_consumo.csv` — período: **2017-01-01 a 2025-05-01**; meses: **101**
- `dados/previsoes_diarias_baselines.csv` — período: **2017-01-01 a 2025-05-31**; linhas: **3073**

## Artefatos gerados por tarefa

### T1 — Médias móveis diárias por ano

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

### T3 — Interrupções e temperatura semanal por ano

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

### T4 — Precipitação semanal e mensal

**Pasta:** `graficos/T4_precipitacao/`

- `interrupcoes_vs_precipitacao_mensal.png`
- `interrupcoes_vs_precipitacao_semanal.png`

### T5 — Dispersões mensais com regressão

**Pasta:** `graficos/T5_scatter_regressao/`

- `scatter_consumo_vs_interrupcoes_gradiente_tempo.png`
- `scatter_consumo_vs_interrupcoes_por_ano.png`
- `scatter_temperatura_vs_consumo_gradiente_tempo.png`
- `scatter_temperatura_vs_consumo_por_ano.png`

### T6 — Baselines no período de teste

**Pasta:** `graficos/T6_previsao_zoom_1ano/`

- `previsao_baselines_teste_2023.png`
- `previsao_baselines_teste_2024.png`
- `previsao_baselines_teste_2025.png`

### T8 — Vento diário integrado

**Pasta:** `graficos/T8_vento/`

- `diario_interrupcoes_vs_rajada_max.png`
- `diario_interrupcoes_vs_vento_vel_media.png`

### T9 — Vento agregado semanal e mensal

**Pasta:** `graficos/T9_vento_agregados/`

- `mensal_interrupcoes_vs_direcao_media.png`
- `mensal_interrupcoes_vs_rajada_max.png`
- `mensal_interrupcoes_vs_vento_vel_media.png`
- `semanal_interrupcoes_vs_direcao_media.png`
- `semanal_interrupcoes_vs_rajada_max.png`
- `semanal_interrupcoes_vs_vento_vel_media.png`

## Correlações de Pearson — resumo canônico

### Maiores correlações em módulo

| nivel_temporal | variavel              | pearson_r           | rotulo                     |
| -------------- | --------------------- | ------------------- | -------------------------- |
| mensal         | vento_dir_cos         | 0.5699785186785606  | Direção do vento (cosseno) |
| mensal         | precipitacao_total_mm | 0.5385577719045647  | Precipitação total         |
| mensal         | vento_dir_sin         | -0.5221782839437505 | Direção do vento (seno)    |
| semanal        | precipitacao_total_mm | 0.4952413780711646  | Precipitação total         |
| mensal         | consumo_total_kwh     | 0.4763950046888337  | Consumo total              |
| mensal         | vento_rajada_max_ms   | 0.4486054693529648  | Rajada máxima              |
| semanal        | vento_dir_sin         | -0.436664602035649  | Direção do vento (seno)    |
| semanal        | vento_rajada_max_ms   | 0.4013909684966619  | Rajada máxima              |
| semanal        | vento_dir_cos         | 0.3948139680897146  | Direção do vento (cosseno) |
| mensal         | temperatura_media     | 0.3937289450527473  | Temperatura média          |
| diario         | precipitacao_total_mm | 0.3476214074842246  | Precipitação total         |
| diario         | vento_dir_sin         | -0.3453591317833634 | Direção do vento (seno)    |

### Tabela completa

| nivel_temporal | variavel                  | pearson_r           | rotulo                     |
| -------------- | ------------------------- | ------------------- | -------------------------- |
| diario         | temperatura_media         | 0.1018850744761465  | Temperatura média          |
| diario         | precipitacao_total_mm     | 0.3476214074842246  | Precipitação total         |
| diario         | vento_velocidade_media_ms | -0.1490545642273023 | Velocidade média do vento  |
| diario         | vento_velocidade_max_ms   | 0.0335751571648983  | Velocidade máxima do vento |
| diario         | vento_rajada_max_ms       | 0.2544010793657216  | Rajada máxima              |
| diario         | vento_dir_sin             | -0.3453591317833634 | Direção do vento (seno)    |
| diario         | vento_dir_cos             | 0.2102742973201247  | Direção do vento (cosseno) |
| semanal        | temperatura_media         | 0.2508928603096201  | Temperatura média          |
| semanal        | precipitacao_total_mm     | 0.4952413780711646  | Precipitação total         |
| semanal        | vento_velocidade_media_ms | -0.2497614185844047 | Velocidade média do vento  |
| semanal        | vento_velocidade_max_ms   | 0.0723629493030534  | Velocidade máxima do vento |
| semanal        | vento_rajada_max_ms       | 0.4013909684966619  | Rajada máxima              |
| semanal        | vento_dir_sin             | -0.436664602035649  | Direção do vento (seno)    |
| semanal        | vento_dir_cos             | 0.3948139680897146  | Direção do vento (cosseno) |
| mensal         | temperatura_media         | 0.3937289450527473  | Temperatura média          |
| mensal         | precipitacao_total_mm     | 0.5385577719045647  | Precipitação total         |
| mensal         | vento_velocidade_media_ms | -0.3102246509225288 | Velocidade média do vento  |
| mensal         | vento_velocidade_max_ms   | 0.0695457408517114  | Velocidade máxima do vento |
| mensal         | vento_rajada_max_ms       | 0.4486054693529648  | Rajada máxima              |
| mensal         | vento_dir_sin             | -0.5221782839437505 | Direção do vento (seno)    |
| mensal         | vento_dir_cos             | 0.5699785186785606  | Direção do vento (cosseno) |
| mensal         | consumo_total_kwh         | 0.4763950046888337  | Consumo total              |

## Interpretação resumida dos achados

- A agregação temporal não fortalece todas as relações, mas evidencia alguns padrões acumulados.
- A precipitação apresenta correlação de aproximadamente **0,348** no diário, **0,495** no semanal e **0,539** no mensal.
- No nível mensal, interrupções e consumo têm correlação de aproximadamente **0,476**.
- A direção do vento é circular: seus componentes seno e cosseno alcançam, no mensal, aproximadamente **-0,522** e **0,570**, respectivamente.
- Correlação descreve associação e, isoladamente, não demonstra causalidade.

## Próximos passos sugeridos

- Comparar modelos preditivos por divisão temporal e validação walk-forward, sempre evitando vazamento.
- Documentar janelas de atributos, hiperparâmetros e métricas de teste.
- Manter os artefatos históricos sincronizados com a base e as regras canônicas do projeto.
