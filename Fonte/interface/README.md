# Interface Streamlit

Interface local para explorar os dados, treinar os modelos e visualizar os
resultados do TCC sem editar ou executar comandos Python individualmente.

## Início rápido no Windows

Na raiz do projeto:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run Fonte\interface\app.py
```

Também é possível clicar duas vezes em `iniciar_interface.bat` depois de
instalar as dependências.

## Funcionalidades

- painel exploratório da série diária;
- página própria para a previsão direta do próximo dia (`t → t+1`);
- treinamento direto por faixa de datas;
- modo multi-horizonte com modelos independentes para 1, 3, 7 e 14 dias;
- comparação de métricas globais e mês a mês;
- XGBoost, Bi-LSTM, Bi-GRU e ARIMAX experimental;
- exportação de previsões e métricas em CSV;
- execução controlada dos scripts do projeto.

Cada treinamento iniciado pela interface cria uma pasta em
`Fonte/results/interface/`, contendo:

- `configuracao.json`;
- `metricas.csv`;
- `previsoes.csv`.

## Observação metodológica

Na interface, **previsão direta** significa prever o próximo dia (`h = 1`). O modo **multi-horizonte** treina uma tarefa direta independente para cada horizonte selecionado, sem previsão recursiva.

Os períodos são separados pela **data-alvo**. Nenhuma data usada para avaliar
o modelo participa do ajuste. A janela histórica das redes neurais fornece
contexto anterior à primeira data-alvo selecionada.

O ARIMAX foi incluído como comparação experimental solicitada pelo orientador.
Ele recebe covariáveis observadas na data de origem. Para uso operacional,
essas covariáveis precisariam ser substituídas por previsões meteorológicas
disponíveis no instante da decisão.
