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

1. **Dados integrados**: INMET (meteorologia), ANEEL (interrupções) e SAMP/CCEE (consumo energético), totalizando 3.073 dias brutos e 3.058 dias após engenharia de atributos.

2. **Engenharia de atributos**: pipeline com 41 features derivadas — defasagens de 1 a 7 dias, médias móveis exponenciais (spans 3, 7 e 14 dias), desvio-padrão móvel e codificações harmônicas cíclicas de calendário.

3. **Modelos comparados** sob protocolo de separação temporal estrita (*Out-of-Sample*):
   - **XGBoost** com Grid Search temporal (5 folds, 7 hiperparâmetros)
   - **Bi-LSTM** em PyTorch (2 camadas, hidden=64, Dropout=0.4, AdamW)
   - **Bi-GRU** em PyTorch (mesma arquitetura da Bi-LSTM)

4. **Avaliação multi-horizonte**: os três modelos foram avaliados em horizontes de previsão de 1, 3, 7 e 14 dias, usando previsão direta (XGBoost) e estratégia recursiva auto-regressiva (Bi-LSTM e Bi-GRU).

---

## Resultados

### Avaliação padrão (365 dias de teste)

| Modelo | MAE | RMSE | R² | MAPE |
|:---|:---:|:---:|:---:|:---:|
| **XGBoost** | 52.31 | 89.85 | 0.520 | 16.06% |
| Bi-LSTM | 59.30 | 100.07 | 0.410 | 18.83% |
| Bi-GRU | 65.68 | 106.55 | 0.332 | 21.19% |

### Avaliação multi-horizonte (MAE por dias à frente)

| Horizonte | XGBoost | Bi-LSTM | Bi-GRU |
|:---:|:---:|:---:|:---:|
| 1 dia | 70.6 | **29.4** | 62.4 |
| 3 dias | **68.2** | 73.6 | 82.1 |
| 7 dias | **46.6** | 54.1 | 59.2 |
| 14 dias | **40.6** | 44.2 | 50.4 |

> **Achado principal**: a Bi-LSTM supera o XGBoost em previsão de 1 dia à frente (inversão de hierarquia). A partir de 3 dias, o XGBoost retoma e mantém a liderança — efeito da propagação de erros na estratégia recursiva das redes neurais.

---

## Estrutura

```
TCC/
├── Monografia/              # Texto integral em LaTeX (146 páginas)
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
│   │       ├── plot_multihorizonte.py       # gráficos de degradação
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

Consulte [`Fonte/README.md`](Fonte/README.md) para instruções detalhadas de instalação, reprodução e mapeamento de scripts.

```bash
cd Fonte/src/models

python baseline_xgboost.py          # XGBoost (~1 min CPU)
python lstm_bidirecional.py         # Bi-LSTM (~5-10 min CPU/GPU)
python gru_avancada.py              # Bi-GRU  (~5-10 min CPU/GPU)
python previsao_multihorizonte.py   # análise multi-horizonte
```

**Reprodutibilidade**: XGBoost é determinístico (`random_state=42`). PyTorch pode apresentar pequenas variações entre execuções por non-determinism do cuDNN.

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
