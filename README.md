<h1 align="center">
  <br>
  <img src="Monografia/img/unb_logo.png" alt="UnB Logo" width="120">
  <br>
  Modelagem Preditiva de Falhas na Rede Elétrica do Distrito Federal
  <br>
</h1>

<h4 align="center">Um Estudo Comparativo entre XGBoost e Redes Neurais Recorrentes Bidirecionais (Bi-LSTM / Bi-GRU)</h4>

<p align="center">
  Giovanni Minari Zanetti &nbsp;•&nbsp; Mateus Gomes de Araújo<br>
  Orientador: Prof. Jan Mendonça Corrêa (CIC/UnB) &nbsp;•&nbsp; Brasília, 2026
</p>

<p align="center">
  <a href="#sobre">Sobre</a> •
  <a href="#metodologia">Metodologia</a> •
  <a href="#resultados">Resultados</a> •
  <a href="#estrutura">Estrutura</a> •
  <a href="#como-executar">Como Executar</a> •
  <a href="#tecnologias">Tecnologias</a>
</p>

---

## Sobre

Este repositório contém o código-fonte, os dados processados e o texto integral em LaTeX do TCC em Ciência da Computação defendido na Universidade de Brasília (UnB).

O trabalho desenvolve e avalia modelos para estimar o número diário total de interrupções no Distrito Federal, integrando dados governamentais ao longo de **3.073 dias consecutivos (2017–2025)**. O alvo não foi filtrado por causa e contém **748.542 ocorrências deduplicadas**.

---

## Metodologia

1. **Dados integrados**: INMET (meteorologia), ANEEL (interrupções) e SAMP/ANEEL (consumo energético exploratório), totalizando 3.073 dias brutos e 3.066 dias após engenharia de atributos.

2. **Engenharia de atributos**: 40 variáveis de entrada, sete meteorológicas base e 33 derivadas — defasagens em 1, 2, 3 e 7 dias, EMAs, desvio-padrão móvel, calendário cíclico e direção do vento em seno/cosseno.

3. **Modelos comparados** sob protocolo de separação temporal estrita (*Out-of-Sample*, últimos 365 dias):
   - **XGBoost** com Grid Search temporal (TimeSeriesSplit, 5 folds, 7 hiperparâmetros). Best: max_depth=4, η=0.03, n_estimators=300, subsample=0.7, colsample_bytree=0.8, min_child_weight=1
   - **Bi-LSTM** em PyTorch (2 camadas, hidden=64, Dropout=0.4, AdamW, 150 épocas, semente fixa)
   - **Bi-GRU** em PyTorch (mesma arquitetura da Bi-LSTM)

4. **Avaliação multi-horizonte direta**: modelos independentes para h ∈ {1, 3, 7, 14}, sem recursão, com mesmo corte de origem e 352 datas-alvo idênticas por combinação.

---

## Resultados

### Avaliação principal (365 dias de teste, h=1 direto — mesmas datas-alvo e mesmo corte causal)

| Modelo | MAE | RMSE | R² | MAPE |
|:---|:---:|:---:|:---:|:---:|
| XGBoost | 63.81 | 101.77 | 0.385 | 21.34% |
| **Bi-LSTM** | 61.55 | **98.70** | **0.421** | 19.68% |
| **Bi-GRU** | **60.36** | 102.57 | 0.375 | **18.89%** |
| Persistência | 68.59 | 105.61 | 0.337 | 22.37% |

### Avaliação multi-horizonte com previsão direta (MAE, 352 alvos por horizonte)

| Horizonte | XGBoost | Bi-LSTM | Bi-GRU |
|:---:|:---:|:---:|:---:|
| h=1 dia | 65.27 | 62.92 | **61.67** |
| h=3 dias | 77.22 | **65.56** | 66.52 |
| h=7 dias | 77.64 | 71.86 | **71.24** |
| h=14 dias | 70.41 | 71.21 | **68.19** |

O baseline ingênuo de persistência obteve MAE de 69.83, 86.22, 84.56 e 88.00 nos horizontes de 1, 3, 7 e 14 dias, respectivamente.

> **Achados principais**: a Bi-GRU tem o menor MAE agregado em h=1, h=7 e h=14; a Bi-LSTM lidera em h=3. A avaliação de severidade contém 73 dias Normais, 242 Moderados e 50 Severos, com aumento claro do erro nos maiores valores.

---

## Estrutura

```
TCC/
├── Monografia/              # Texto integral em LaTeX (PDF compilado)
│   ├── tex/                 # Capítulos 1 a 6 + apêndices
│   ├── img/                 # Figuras geradas em Python
│   ├── monografia.tex       # Entrypoint (classe UnB-CIC)
│   └── monografia.pdf       # PDF final compilado
│
├── Fonte/                   # Código-fonte e dados
│   ├── data/                # CSVs processados
│   ├── run_pipeline.py      # reprodução integral em um comando
│   ├── src/                 # Scripts Python
│   │   ├── build_base_from_raw.py
│   │   ├── 01_eda_sazonalidade.py
│   │   ├── 02_correlacoes_nao_lineares.py
│   │   ├── 03_feature_engineering.py
│   │   ├── 04_eda_basica.py
│   │   ├── 05_correlacoes_unificadas.py
│   │   └── models/
│   │       ├── baseline_xgboost.py
│   │       ├── baseline_persistence.py
│   │       ├── lstm_bidirecional.py
│   │       ├── gru_avancada.py
│   │       ├── evaluate_severity.py
│   │       ├── previsao_multihorizonte.py   # avaliação multi-horizonte
│   │       ├── plot_multihorizonte.py       # gráficos de desempenho por horizonte
│   │       └── advanced_plots.py
│   └── results/             # Gráficos e métricas gerados
│       ├── eda/
│       └── ml/
│
├── RespostasJan/            # Respostas às dúvidas do orientador
└── ROTEIRO_PROFESSOR.md     # Mapeamento figura/tabela → script gerador
```

---

## Como Executar

### Interface gráfica

A interface permite explorar os dados, treinar os modelos por período, comparar
meses e horizontes e executar as análises sem editar os scripts:

```bash
python -m streamlit run Fonte/interface/app.py
```

No Windows, depois de instalar as dependências, também é possível abrir
`iniciar_interface.bat`. Consulte [`Fonte/interface/README.md`](Fonte/interface/README.md)
para o guia completo.

Consulte [`Fonte/README.md`](Fonte/README.md) para instruções detalhadas de instalação, reprodução e mapeamento de scripts.

A reprodução integral parte dos arquivos brutos, constrói o alvo total sem filtro por causa, treina os modelos, gera métricas/figuras e sincroniza as imagens da monografia:

```bash
python Fonte/run_pipeline.py \
  --interruptions D:/dados/aneel/dados_completos_brasilia.csv \
  --inmet-dir D:/dados/inmet
```

Arquivos esperados:

- ANEEL: `dados_completos_brasilia.csv`, SHA-256 `9b68feaad48bdf50d7f8e645d576efc2ccdfecf4aa43672ece3dc771fab905be`.
- INMET: nove CSVs anuais da estação A001, de 2017 a 2025; nomes e hashes estão em `Fonte/data/manifesto_dados_brutos.json`.

Período efetivo: `2017-01-01` a `2025-05-31`. A definição do alvo e a ausência de filtro por causa ficam registradas em `Fonte/data/manifesto_dados_brutos.json`. As versões exatas das bibliotecas estão em `requirements.txt`.

**Reprodutibilidade**: sementes e opções determinísticas são fixadas. Em hardware, drivers ou versões diferentes, bibliotecas numéricas e CUDA ainda podem produzir pequenas variações de ponto flutuante; por isso, o ambiente e os hashes precisam acompanhar os resultados.

---

## Tecnologias

<div align="center">
  <img width="55" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" alt="Python" title="Python"/> &nbsp;&nbsp;
  <img width="50" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" alt="PyTorch" title="PyTorch"/> &nbsp;&nbsp;
  <img width="50" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" alt="Pandas" title="Pandas"/> &nbsp;&nbsp;
  <img width="50" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" alt="NumPy" title="NumPy"/> &nbsp;&nbsp;
  <img width="50" src="https://raw.githubusercontent.com/dmlc/dmlc.github.io/master/img/logo-m/xgboost.png" alt="XGBoost" title="XGBoost"/> &nbsp;&nbsp;
  <br><br>
  <img width="50" src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/latex/latex-original.svg" alt="LaTeX" title="LaTeX"/> &nbsp;&nbsp;
  <img width="65" src="Monografia/img/scikit_logo.svg" alt="Scikit-Learn" title="Scikit-Learn"/>
</div>

<br>

<p align="center">
  Universidade de Brasília — Departamento de Ciência da Computação, 2026
</p>
