# Código-Fonte e Dados — TCC

> **Predição de Interrupções no Fornecimento de Energia Elétrica no Distrito Federal: Um Estudo Comparativo entre XGBoost e Redes Neurais Recorrentes Bidirecionais**
>
> Giovanni Minari Zanetti, Mateus Gomes de Araújo
> Orientador: Prof. Jan Mendonça Corrêa (CIC/UnB)
> Brasília, 2026

Este pacote contém o código-fonte e os dados processados utilizados na monografia. Para o mapeamento detalhado **figura/tabela ↔ script gerador**, consulte o arquivo `ROTEIRO_PROFESSOR.md` na raiz do TCC.

---

## Estrutura

```
Fonte/
├── data/                                  # Dados processados (CSV)
│   ├── base_diaria_interrupcoes_clima.csv
│   ├── base_diaria_interrupcoes_clima_vento.csv
│   ├── base_diaria_interrupcoes_clima_mm.csv
│   ├── base_mensal_interrupcoes_clima_consumo.csv
│   ├── dataset_engenharia_features.csv     # ← saída do script 03
│   ├── agregados_*_canonicos.csv           # agregados semanais/mensais finais
│   ├── correlacoes_consolidadas.csv        # tabela canônica
│   ├── manifesto_dados_brutos.json         # nomes, período e SHA-256
│   ├── legado/                            # artefatos de versões anteriores (não usados pelos modelos finais)
│   │   ├── metricas_dl_lstm_gru.csv
│   │   ├── previsoes_dl_lstm_gru.csv
│   │   └── previsoes_diarias_baselines.csv
│   └── vento_diario_brasilia.csv
│
├── run_pipeline.py                         # execução reproduzível integral
├── src/                                    # Pipeline principal
│   ├── build_base_from_raw.py              # bruto ANEEL/INMET → base diária
│   ├── 01_eda_sazonalidade.py             # decomposição STL + ACF/PACF
│   ├── 02_correlacoes_nao_lineares.py     # Spearman/Kendall + cross-corr
│   ├── 03_feature_engineering.py          # gera dataset_engenharia_features.csv
│   ├── 04_eda_basica.py                    # série completa, distribuição, etc.
│   ├── 05_correlacoes_unificadas.py        # agregações/correlações canônicas
│   └── models/
│       ├── data_loader_dl.py              # janelamento + MinMax para PyTorch
│       ├── baseline_xgboost.py            # XGBoost: treino + avaliação + gráficos
│       ├── lstm_bidirecional.py           # Bi-LSTM: treino + avaliação + gráficos
│       ├── gru_avancada.py                # Bi-GRU: treino + avaliação + gráficos
│       ├── evaluate_severity.py            # avaliação por faixa, y_true único
│       ├── advanced_plots.py              # gráficos comparativos (KDE, heteroscedasticidade)
│       └── script_exploration_pipeline.py # EDA exploratória rápida (3 figuras)
│
├── results/                                # Artefatos gerados pelos scripts
│   ├── eda/                                # gráficos exploratórios
│   └── ml/                                 # gráficos e métricas dos modelos
```

Scripts adicionais que produzem figuras temáticas estão em `../SegundoPedido/SegundoPedido/scripts/` e `../TerceiroPedido/scripts/`. O `ROTEIRO_PROFESSOR.md` lista cada um.

---

## Pipeline de dados (do bruto ao dataset final)

A reprodução usa os arquivos brutos oficiais já baixados pelo pesquisador:

- **ANEEL**: `dados_completos_brasilia.csv`, SHA-256 `9b68feaad48bdf50d7f8e645d576efc2ccdfecf4aa43672ece3dc771fab905be`.
- **INMET**: nove arquivos `INMET_CO_DF_A001_BRASILIA_*.CSV`, um por ano de 2017 a 2025. Os nomes e SHA-256 individuais estão em `data/manifesto_dados_brutos.json`.
- **Período usado**: 2017-01-01 a 2025-05-31.
- **SAMP/ANEEL**: consumo mensal já consolidado, usado apenas em correlações exploratórias e nunca como entrada dos modelos.

Nenhum filtro por causa é aplicado aos registros da ANEEL. O alvo corresponde ao total diário de interrupções após a deduplicação por `NumOrdemInterrupcao`: 748.542 eventos únicos no período. Essa definição e `filtro_por_causa: false` ficam registrados em `data/manifesto_dados_brutos.json`.

A engenharia de atributos (`src/03_feature_engineering.py`) gera:

   - Calendário: `mes`, `dia_semana`, `dia_ano`, `mes_sin`, `mes_cos`
   - Defasagens (lags): 1, 2, 3 e 7 dias para `interrupcoes`, `precipitacao_total_mm`, `temperatura_media`, `vento_rajada_max_ms`
   - Médias móveis exponenciais (EMA): spans 3, 7 e 14 dias para chuva, temperatura e rajada
   - Desvio-padrão móvel (rolling std) de 7 dias para chuva, temperatura e rajada
    - Direção do vento representada circularmente por `vento_dir_sin` e `vento_dir_cos`
    - Total final: **3.066 dias × 40 variáveis de entrada**, sete base e 33 derivadas, além do canal histórico do alvo disponível na origem.

As agregações canônicas usam soma para interrupções e precipitação, média para temperatura/velocidade, máximo para rajada/máximas e componentes circulares para direção.

---

## Requisitos

- Python ≥ 3.10 (testado com 3.14)
- Bibliotecas instaláveis via `pip install -r ../requirements.txt`:
  - pandas, numpy, scikit-learn ≥ 1.4, xgboost ≥ 2.0, torch ≥ 2.0,
    matplotlib, seaborn, statsmodels.

## Como executar

A partir da raiz do repositório, com os dados brutos em diretórios separados:

```bash
python Fonte/run_pipeline.py \
  --interruptions D:/dados/aneel/dados_completos_brasilia.csv \
  --inmet-dir D:/dados/inmet
```

O comando:

1. arquiva os resultados anteriores em `results/archive/`;
2. reconstrói a base diária e os hashes;
3. aplica a engenharia de atributos e as agregações canônicas;
4. executa testes automatizados;
5. treina XGBoost, Bi-LSTM e Bi-GRU;
6. avalia severidade e horizontes 1, 3, 7 e 14;
7. gera diagnósticos e sincroniza as figuras com `Monografia/img/`.

## Reprodutibilidade

- Versões exatas: `../requirements.txt`.
- Integridade dos brutos: `data/manifesto_dados_brutos.json`.
- Definição do alvo: `data/manifesto_dados_brutos.json` registra o total diário e a ausência de filtro por causa.
- Testes: `python -m unittest discover -s Fonte/tests -p "test_*.py" -q`.
- As sementes e opções determinísticas são fixadas. Hardware, driver, CUDA ou versões diferentes ainda podem produzir pequenas variações numéricas.

## Métricas de referência (test set, 365 dias, h=1 direto)

Todos os modelos preveem `interrupcoes[t+1]` com as mesmas datas-alvo e sem informação futura. O MAPE exclui apenas observações com `y_true=0`; no teste final não há zeros.

| Modelo | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| XGBoost | 63,81 | 101,77 | 0,385 | 21,34% |
| **Bi-LSTM** | 61,55 | **98,70** | **0,421** | 19,68% |
| **Bi-GRU** | **60,36** | 102,57 | 0,375 | **18,89%** |
| Persistência | 68,59 | 105,61 | 0,337 | 22,37% |
