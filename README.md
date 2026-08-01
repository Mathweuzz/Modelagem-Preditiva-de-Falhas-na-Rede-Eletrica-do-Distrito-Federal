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

O trabalho desenvolve e avalia modelos preditivos para estimar o número diário de interrupções no fornecimento de energia elétrica no Distrito Federal, integrando dados de três fontes governamentais abertas ao longo de **3.073 dias consecutivos (2017–2025)**. O objetivo é fornecer à concessionária uma ferramenta de apoio à decisão para pré-posicionamento de equipes de manutenção antes de eventos climáticos adversos.

---

## Metodologia

1. **Dados integrados**: INMET (meteorologia), ANEEL (interrupções) e SAMP/CCEE (consumo energético), totalizando 3.073 dias brutos e 3.066 dias após engenharia de atributos (lacunas temporais tratadas por interpolação linear).

2. **Engenharia de atributos**: pipeline com 40 features derivadas — defasagens de 1 a 7 dias, médias móveis exponenciais (spans 3, 7 e 14 dias), desvio-padrão móvel e codificações harmônicas cíclicas de calendário.

3. **Modelos comparados** sob protocolo de separação temporal estrita (*Out-of-Sample*, últimos 365 dias):
   - **XGBoost** com Grid Search temporal (TimeSeriesSplit, 5 folds, 7 hiperparâmetros). Best: max_depth=4, η=0.03, n_estimators=300, subsample=0.7, colsample_bytree=0.8, min_child_weight=1
   - **Bi-LSTM** em PyTorch (2 camadas, hidden=64, Dropout=0.4, AdamW, 150 épocas, semente fixa)
   - **Bi-GRU** em PyTorch (mesma arquitetura da Bi-LSTM)

4. **Avaliação multi-horizonte com previsão direta**: modelos independentes treinados para cada horizonte h ∈ {1, 3, 7, 14}, sem recursão — comparação equânime com 365 datas-alvo idênticas por modelo e horizonte.

---

## Resultados

### Avaliação principal (365 dias de teste, h=1 direto — todos os modelos com mesma informação)

| Modelo | MAE | RMSE | R² | MAPE |
|:---|:---:|:---:|:---:|:---:|
| **XGBoost** | **61.09** | 99.72 | 0.409 | 20.64% |
| **Bi-LSTM** | 62.71 | **98.80** | **0.420** | 20.94% |
| Bi-GRU | 73.44 | 119.21 | 0.156 | 26.18% |

### Avaliação multi-horizonte com previsão direta (MAE, 365 alvos por horizonte)

| Horizonte | XGBoost | Bi-LSTM | Bi-GRU |
|:---:|:---:|:---:|:---:|
| h=1 dia | 61.09 | **62.71** | 73.44 |
| h=3 dias | 76.71 | **72.08** | 75.78 |
| h=7 dias | 81.57 | 77.70 | **71.81** |
| h=14 dias | 69.50 | 66.12 | **62.79** |

> **Achados principais**: (1) Em h=1, XGBoost e Bi-LSTM empatam tecnicamente (diferença <2%); (2) Para h≥3, as redes recorrentes superam o XGBoost: Bi-LSTM lidera em h=3, Bi-GRU lidera em h=7 e h=14. O XGBoost registra R²=-0.001 em h=7 (sem poder preditivo).

---

## Estrutura

```
TCC/
├── Monografia/              # Texto integral em LaTeX (148 páginas)
│   ├── tex/                 # Capítulos 1 a 6 + apêndices
│   ├── img/                 # Figuras geradas em Python
│   ├── monografia.tex       # Entrypoint (classe UnB-CIC)
│   └── monografia.pdf       # PDF final compilado
│
├── Fonte/                   # Código-fonte e dados
│   ├── data/                # CSVs processados
│   ├── src/                 # Scripts Python
│   │   ├── 01_eda_sazonalidade.py
│   │   ├── 02_correlacoes_nao_lineares.py
│   │   ├── 03_feature_engineering.py
│   │   ├── 04_eda_basica.py
│   │   └── models/
│   │       ├── baseline_xgboost.py
│   │       ├── lstm_bidirecional.py
│   │       ├── gru_avancada.py
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

```bash
cd Fonte/src/models

python baseline_xgboost.py          # XGBoost (~1 min CPU)
python lstm_bidirecional.py         # Bi-LSTM (~5-10 min CPU/GPU)
python gru_avancada.py              # Bi-GRU  (~5-10 min CPU/GPU)
python previsao_multihorizonte.py   # análise multi-horizonte
python plot_multihorizonte_temporal.py # comparação temporal e mensal
```

**Reprodutibilidade**: XGBoost (`random_state=42`) é determinístico. PyTorch (seed=42, `cudnn.deterministic=True`) reproduz os resultados dentro da tolerância de ponto flutuante em hardware e versão idênticos; variações numéricas pequenas são esperadas em GPU/CPU ou versões diferentes.

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
