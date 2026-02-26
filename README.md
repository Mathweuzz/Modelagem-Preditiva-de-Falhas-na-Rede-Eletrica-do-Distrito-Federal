<h1 align="center">
  <br>
  <img src="Monografia/img/unb_logo.png" alt="UnB Logo" width="120">
  <br>
  Modelagem Preditiva de Falhas na Rede Elétrica do Distrito Federal
  <br>
</h1>

<h4 align="center">Um Estudo Comparativo entre XGBoost e Redes Neurais Recorrentes Bidirecionais (Bi-LSTM / Bi-GRU) sob Estresse Hidrometeorológico.</h4>

<p align="center">
  <a href="#-sobre-o-projeto">Sobre</a> •
  <a href="#%EF%B8%8F-arquitetura-e-metodologia">Arquitetura</a> •
  <a href="#-tensores-e-datasets">Datasets</a> •
  <a href="#-resultados-preditivos">Resultados</a> •
  <a href="#-estrutura-do-repositório">Repositório</a> •
  <a href="#-tecnologias">Tecnologias</a>
</p>

---

## ⚡ Sobre o Projeto

Este repositório abriga o código-fonte, a modelagem termodinâmica e o texto integral (LaTeX) do Trabalho de Conclusão de Curso (TCC) em Ciência da Computação defendido na Universidade de Brasília (UnB). 

O trabalho propõe a substituição do paradigma de **Manutenção Reativa Forense** adotado nativamente na *Smart Grid* do Brasil Central por **Defensivas Preventivas Algorítmicas**. Num cruzamento massivo de telemetria meteorológica oficial e despachos punitivos da matriz energética, investigamos o esvaziamento silencioso do ciclo de vida eletromecânico das subestações provocado pelas severas flutuações radiativas e hidrológicas do *El Niño* sobre o Cerrado.

Ao alistar arquiteturas rasas padronizadas (XGBoost) em um duelo balístico contra Unidades Ocultas Bidirecionais Gated (*Bi-GRU/Bi-LSTM*), submetidas via *Backpropagation Through Time* (BPTT), a monografia desvenda matrizes colineares invisíveis unindo rajadas de vento, inércia térmica (*Time Lags*) e a detonação estocástica ininterrupta de transformadores envelhecidos.

---

## 🏗️ Arquitetura e Metodologia

O *pipeline* analítico deste estudo foi desenhado sob pesados axiomas matriciais, fragmentados na extração cíclica bruta dos Diários Oficiais:

1. **Ingestão Autônoma Híbrida**: Fusão (*Merging*) puramente referenciada ao longo de eixos de *Timestamp* UTC entre arquivos diários do Governo (2015-2025).
2. **Engenharia de Tensores (*Feature Engineering*)**: 
    - Extração do Arrasto de Rajada Aerodinâmico (`> 80 km/h`), independentemente da velocidade rotineira do vento.
    - Transformações Trigonométricas ($\sin$ e $\cos$) em ciclos estacionais anuais orbitais para atenuamento de saltos temporais de Dezembro à Janeiro.
3. **Mecânica das *Sliding Windows* (Janelas Climatológicas Deslizantes)**:
   - Expansão da regressão estática na construção dimensional de blocos pregressos de resiliência $t-1$, $t-3$, $t-7$ e o crucial $t-14$, encapsulando o esvaziamento da integridade da resina isoladora da malha devido à Ondas de Calor estacionárias.
4. **Descompasso Estocástico Out-Of-Sample**: Teste implacável limitando o treinamento aos hiatos do *La Niña* pré-2023, forçando generalização "cega" dos algoritmos sobre a ebulição violenta do *El Niño* tardio na janela inferencial.

---

## 📊 Tensores e Datasets

Os *DataFrames* consolidados totalizaram 3.073 dias de vetores ininterruptos cruzando as seguintes APIs Abertas Estatais:

| Fonte Operacional | Classe do Parâmetro Dimensional | Tipo Estocástico |
|:---:|:---|:---:|
| **ANEEL** (Interrupções de Energia Elétrica) | *Target $Y$*. Falhas mecânicas comissionadas sem interferência antrópica (Manutenção/Furto de Condutores deletados). Foco isolado no fato gerador *Descargas Atmosféricas/Árvores*. | Variável Dependente |
| **INMET** (Dados Históricos A001) | Matriz Ambiental $X$. Termodinâmica Seca ($^\circ\text{C}$), Precipitação ($\text{mm}$) e Cinemática Eólica ($\text{m/s}$). | Variável Independente |
| **ANEEL** (SAMP - Balanço) | Perfil transversal do tracionamento diário metropolitano ($\text{MWh}$) impulsionando correntes sub-transientes na malha. | *Feature* de Carga |

---

## 📈 Resultados Preditivos

A modelagem determinística comprovou que o XGBoost — a despeito das penalidades Jacobianas Newtonianas aplicadas via L1/L2 foliar — sucumbe sob a extremidade do "ruído térmico" convectivo. Em dia de céu límpido o aprendizado base por árvores atende demandas estáveis (*Underfitting* homoscedástico), todavia subestima colapsos tempestuosos massivos (*Extremos Outliers*). 

Por preceito inverso, as **Redes Neurais Bi-LSTM e Bi-GRU**, instanciadas na base estrutural do *PyTorch*, valeram-se majestosamente de seu portão de esquecimento (*Forget Gate $f_t$*). Ao reter a memória estrutural do vento de dias anteriores e blindarem falsas inferências por calmaria pontual matutina, garantiram aderência formidável em prever quedas, solidificando as séries temporais numa topologia inquebrável por distorções de horizonte.

---

## 📁 Estrutura do Repositório

Organização modular para replicabilidade matemática impecável por futuros pesquisadores acadêmicos.

```text
📦 TCC
 ┣ 📂 Fonte                 # Source Code (Extração e Inteligência Artificial Python)
 ┃ ┣ 📂 data/               # Repositórios CSV Governamentais limpos
 ┃ ┣ 📂 notebooks/          # Kernel Jupyter (Análise Exploratória EDA)
 ┃ ┗ 📂 src/                # Motor Preditivo (Scripts XGBoost, Bi-LSTM, Bi-GRU)
 ┃
 ┣ 📂 Monografia            # TeX Dist (Código-Fonte integral do Documento ABNT)
 ┃ ┣ 📂 img/                # Pipeline de Plotagem Hiper-Matricial (TiKz e Matplotlib)
 ┃ ┣ 📂 tex/                # Capítulos Modulares da Dissertação (1 ao 6)
 ┃ ┣ 📜 bibliografia.bib    # >28 Referenciais Acadêmicos Nativos (Nature, IEEE, Oxford)
 ┃ ┣ 📜 monografia.tex      # Entrypoint (Classe UnB-CIC, Makeglossaries)
 ┃ ┗ 📜 monografia.pdf      # A Definitiva Documentação de 76 Páginas
 ┃
 ┣ 📂 TerceiroPedido        # Artefatos da Extração, Consolidação Visual e CSVs auxiliares
 ┗ 📜 README.md             # Documentação Global
```

---

## 💻 Tecnologias Empregadas

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

---

<p align="center">
 Desenvolvido com profunda dedicação algorítmica e rigor ciêntifico.
</p>
