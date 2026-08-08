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
│   ├── manifesto_fontes_padronizadas.json  # nomes e SHA-256 das entradas
│   ├── aggregados_*.csv                    # agregados semanais/mensais
│   ├── correlacoes_*.csv
│   ├── legado/                            # artefatos de versões anteriores (não usados pelos modelos finais)
│   │   ├── metricas_dl_lstm_gru.csv
│   │   ├── previsoes_dl_lstm_gru.csv
│   │   └── previsoes_diarias_baselines.csv
│   └── vento_diario_brasilia.csv
│
├── src/                                    # Pipeline principal
│   ├── build_base_from_standardized.py     # fontes padronizadas → base diária
│   ├── 01_eda_sazonalidade.py             # decomposição STL + ACF/PACF
│   ├── 02_correlacoes_nao_lineares.py     # Spearman/Kendall + cross-corr
│   ├── 03_feature_engineering.py          # gera dataset_engenharia_features.csv
│   ├── 04_eda_basica.py                    # série completa, distribuição, etc.
│   └── models/
│       ├── data_loader_dl.py              # janelamento + MinMax para PyTorch
│       ├── baseline_xgboost.py            # XGBoost: treino + avaliação + gráficos
│       ├── lstm_bidirecional.py           # Bi-LSTM: treino + avaliação + gráficos
│       ├── gru_avancada.py                # Bi-GRU: treino + avaliação + gráficos
│       ├── advanced_plots.py              # gráficos comparativos (KDE, heteroscedasticidade)
│       └── script_exploration_pipeline.py # EDA exploratória rápida (3 figuras)
│
├── results/                                # Artefatos gerados pelos scripts
│   ├── eda/                                # gráficos exploratórios
│   └── ml/                                 # gráficos e métricas dos modelos
│
└── venv/                                   # ambiente Python (local — não distribuir)
```

Scripts adicionais que produzem figuras temáticas estão em `../SegundoPedido/SegundoPedido/scripts/` e `../TerceiroPedido/scripts/`. O `ROTEIRO_PROFESSOR.md` lista cada um.

---

## Pipeline de dados (das fontes padronizadas ao dataset final)

A monografia descreve o fluxo completo no Capítulo 3 (Metodologia). O construtor reproduz a base a partir de cópias oficiais previamente padronizadas. Ele não lê diretamente o formato original baixado dos portais: os arquivos do INMET, por exemplo, ainda exigem remoção do cabeçalho de metadados, conversão do separador e da codificação, normalização dos nomes das colunas e conversão da vírgula decimal.

O processamento possui três estágios:

1. **Coleta** (não automatizada neste pacote — fontes oficiais):
   - **INMET** — Instituto Nacional de Meteorologia: estação automática A001 (Brasília), com dados horários de temperatura, precipitação, vento médio, vento máximo e rajada máxima (2017-01-01 a 2025-05-31).
   - **ANEEL** — Agência Nacional de Energia Elétrica: relatórios de continuidade do PRODIST, indicadores de interrupção da concessionária Neoenergia Brasília.
   - **SAMP/CCEE** — consumo mensal de energia no DF.
2. **Consolidação diária** (`Fonte/src/build_base_from_standardized.py`): seleção exclusiva da estação A001, validação da unicidade por data/hora, agregação horária → diária com estatística circular para a direção do vento e união com a contagem diária de interrupções únicas.
3. **Engenharia de atributos** (`Fonte/src/03_feature_engineering.py`): a partir de `base_diaria_interrupcoes_clima_vento.csv`, gera `dataset_engenharia_features.csv` adicionando:
   - Direção do vento: componentes unitárias `vento_dir_sin` e `vento_dir_cos`, renormalizadas após eventual interpolação
   - Calendário: `mes`, `dia_semana`, `dia_ano`, `mes_sin`, `mes_cos`
   - Defasagens (lags): 1, 2, 3 e 7 dias para `interrupcoes`, `precipitacao_total_mm`, `temperatura_media`, `vento_rajada_max_ms`
   - Médias móveis exponenciais (EMA): spans 3, 7 e 14 dias para chuva, temperatura e rajada
   - Desvio-padrão móvel (rolling std) de 7 dias para chuva, temperatura e rajada
   - Drop de NaNs iniciais e interpolação linear de lacunas temporais. Total final: **3.066 dias × 40 features + alvo**.

O script reproduz a estrutura e os valores do dataset; diferenças numéricas residuais de ponto flutuante são possíveis entre versões de bibliotecas.

---

## Requisitos

- Python ≥ 3.10 (testado com 3.14)
- Bibliotecas instaláveis via `pip install -r ../requirements.txt`:
  - pandas, numpy, scikit-learn ≥ 1.4, xgboost ≥ 2.0, torch ≥ 2.0,
    matplotlib, seaborn, statsmodels.

## Como executar

A reconstrução exige o CSV consolidado da ANEEL e um diretório com os nove arquivos anuais padronizados da estação A001. A partir da raiz do repositório:

```bash
# 1. Base diária reproduzível
Fonte/venv/bin/python Fonte/src/build_base_from_standardized.py \
  --interruptions /caminho/dados_completos_brasilia.csv \
  --inmet-dir /caminho/dados_clima-inmet_limpos \
  --output-dir Fonte/data

# 2. Engenharia de atributos
cd Fonte/src
../venv/bin/python 03_feature_engineering.py

# 3. EDA — Exploração inicial
python 01_eda_sazonalidade.py            # decomposição + ACF/PACF
python 02_correlacoes_nao_lineares.py    # Spearman/Kendall + cross-corr
python 04_eda_basica.py                  # série completa, distribuição, etc.

# 4. Modelos preditivos
cd models
python script_exploration_pipeline.py    # 3 figuras EDA finais
python baseline_xgboost.py               # XGBoost (~1 min CPU)
python baseline_persistence.py           # baseline ingênuo (~1 s)
python lstm_bidirecional.py              # Bi-LSTM (~5-10 min CPU)
python gru_avancada.py                   # Bi-GRU  (~5-10 min CPU)
python previsao_multihorizonte.py        # avaliação multi-horizonte (h=1,3,7,14)
python plot_multihorizonte.py            # gráficos de desempenho por horizonte
python plot_multihorizonte_temporal.py   # séries anuais e MAE mensal por horizonte
python advanced_plots.py                 # gráficos comparativos (KDE, heteroscedasticidade)
```

## Reprodutibilidade

- **Integridade das entradas**: nomes e hashes SHA-256 em `data/manifesto_fontes_padronizadas.json`.
- **Testes**: `Fonte/venv/bin/python -m unittest discover -s Fonte/tests -p "test_*.py" -v`, executado a partir da raiz.
- **XGBoost**: `random_state=42`. Reprodução determinística.
- **LSTM/GRU (PyTorch)**: pequenas variações são esperadas entre execuções por causa do non-determinism interno do cuDNN/CUDA. As métricas reportadas no Capítulo 4 da monografia foram obtidas com o ambiente especificado no Apêndice (Reprodutibilidade).

## Métricas de referência (test set, 365 dias, h=1 direto)

Avaliação principal: todos os modelos preveem interrupcoes[t+1] usando features do dia t.
Métricas atualizadas após re-execução com protocolo corrigido (ver commit mais recente).

| Modelo | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| **XGBoost** | 61,09 | 99,72 | 0,409 | 20,64% |
| **Bi-LSTM** | 62,71 | 98,80 | 0,420 | 20,94% |
| **Bi-GRU**  | 73,44 | 119,21 | 0,156 | 26,18% |
