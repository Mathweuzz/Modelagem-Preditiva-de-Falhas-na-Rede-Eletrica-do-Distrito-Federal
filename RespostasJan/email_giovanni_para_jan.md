# Email — Giovanni para Prof. Jan

**Para:** jan@unb.br *(confirmar endereço)*
**Assunto:** Re: Retorno sobre matrícula, feedback da monografia e perguntas

---

Prezado Prof. Jan,

Obrigado pelo retorno e pelas perguntas — ficamos felizes em esclarecer cada ponto.

**Sobre a Fig 4.1 — período de teste:**
O período de teste corresponde aos últimos 365 dias do dataset, de 01/06/2024 a 31/05/2025. O corte foi feito por número de dias (`test_size_days = 365`), não por data calendário. A linha tracejada na figura marca exatamente esse ponto.

**Sobre a Fig 4.2 — 2024 e 2025 inteiros para teste?**
Não. O ano de 2024 está dividido: janeiro a maio de 2024 pertencem ao treino, e junho de 2024 em diante ao teste. As barras aparecem em vermelho porque a legenda indica "período que inclui dados de teste" — não que o ano inteiro é teste.

**Sobre as Figs 4.20 a 4.22 — divisão treino/teste:**
Os três gráficos mostram somente o conjunto de teste (365 dias: 01/06/2024–31/05/2025). A divisão foi: 2.693 dias de treino (08/01/2017 a 31/05/2024) e 365 dias de teste (01/06/2024 a 31/05/2025), aplicada de forma idêntica nos três modelos.

**Sobre os picos nos valores reais:**
Os picos mais severos do período de teste ocorreram em 10–11/out/2024 (acima de 1.000 interrupções, início da estação chuvosa), 22–23/jan/2025 (rajadas de 15,1 e 18,6 m/s) e 14/mar/2025 (38,4 mm de precipitação em um dia). Os eventos de outubro apresentam leituras moderadas na estação INMET, o que é esperado: tempestades convectivas do cerrado são espacialmente localizadas e podem atingir múltiplas subestações sem gerar leituras extremas em um único ponto de medição.

**Sobre uso de dados de treino no teste:**
Não ocorre vazamento em nenhum modelo. O `MinMaxScaler` é ajustado exclusivamente nos dados de treino (`.fit_transform`) e apenas aplicado ao teste (`.transform`). As features de lag (t-1, t-2, t-3, t-7) e EMA representam informação passada disponível no momento real da previsão — não há uso de valores futuros.

**Sobre o modelo com previsão de 3 dias:**
Sim, seria possível. O que o senhor descreve é uma previsão multi-step com horizonte fixo de 3 dias, diferente do modelo atual que prevê 1 dia por vez. Para o XGBoost, seriam necessários 3 modelos independentes (um por horizonte); para LSTM e GRU, a camada de saída mudaria de 1 para 3 neurônios. Tecnicamente viável, mas implicaria refazer o pipeline de treinamento, avaliação e boa parte do Capítulo 4. Gostaríamos de entender se o senhor vê isso como necessário para a aprovação do trabalho ou se poderia figurar como sugestão de trabalho futuro.

**Sobre localização:**
Os dados utilizados são agregados para todo o DF — o total diário de interrupções em toda a área de concessão da Neoenergia Brasília, sem desagregação por alimentador, subestação ou região administrativa. O modelo prevê esse total consolidado. Uma análise por localização exigiria dados da ANEEL no nível de circuito (SARDH/SAFDI), que não são de acesso público irrestrito e estavam fora do escopo deste trabalho.

Segue em anexo um PDF com as respostas detalhadas, incluindo os trechos de código que comprovam cada ponto.

Qualquer dúvida adicional, estamos à disposição.

Atenciosamente,
Giovanni Minari Zanetti e Mateus Gomes de Araújo
