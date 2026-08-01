# Roteiro de falas — Defesa do TCC

**Trabalho:** Predição de Interrupções no Fornecimento de Energia Elétrica no Distrito Federal  
**Apresentadores:** Giovanni Minari Zanetti e Mateus Gomes de Araújo  
**Tempo estimado da apresentação principal:** aproximadamente 28 minutos  
**Divisão:** Giovanni apresenta os slides 1 a 11; Mateus apresenta os slides 12 a 22; o slide 23 é conduzido pelos dois. Os slides 24 a 27 são de reserva e só devem ser usados se a banca perguntar.

> Este roteiro foi escrito para orientar a fala, e não para ser decorado palavra por palavra. Durante o ensaio, mantenham as ideias centrais, usem um ritmo natural e apontem para os gráficos apenas quando isso ajudar a leitura.

---

## Slide 1 — Predição de Interrupções no Fornecimento de Energia Elétrica no Distrito Federal

**Apresentador:** Giovanni  
**Tempo aproximado:** 30 segundos

**Fala sugerida:**

Bom dia. Eu sou o Giovanni Minari Zanetti e este é o Mateus Gomes de Araújo. Nós vamos apresentar nosso trabalho de conclusão de curso em Engenharia de Computação, desenvolvido sob orientação do professor Jan Mendonça Corrêa. O trabalho investiga a predição de interrupções no fornecimento de energia elétrica no Distrito Federal, comparando o XGBoost com duas redes neurais recorrentes bidirecionais: a Bi-LSTM e a Bi-GRU.

**Transição:** Para começar, vamos apresentar a pergunta que orientou toda a pesquisa.

---

## Slide 2 — Podemos antecipar as interrupções?

**Apresentador:** Giovanni  
**Tempo aproximado:** 1 minuto

**Fala sugerida:**

A pergunta central do trabalho é a seguinte: usando apenas dados públicos de clima e de operação, é possível prever quantas interrupções ocorrerão por dia na rede elétrica do Distrito Federal?

Para responder a essa pergunta, definimos como variável-alvo o número diário de interrupções e comparamos três famílias de modelos: o XGBoost, que trabalha com atributos tabulares, e duas arquiteturas recorrentes, a Bi-LSTM e a Bi-GRU, que aprendem relações temporais em sequências.

Também avaliamos quatro horizontes de previsão: um, três, sete e quatorze dias à frente. Isso é importante porque a utilidade operacional muda conforme o horizonte. Uma previsão para amanhã pode apoiar um alerta imediato, enquanto uma previsão para duas semanas pode ajudar no planejamento de equipes e materiais.

**Transição:** Essa pergunta se torna especialmente relevante quando observamos a evolução histórica das interrupções.

---

## Slide 3 — As interrupções cresceram 65,3% em oito anos

**Apresentador:** Giovanni  
**Tempo aproximado:** 1 minuto e 20 segundos

**Fala sugerida:**

Este gráfico mostra a evolução da média diária de interrupções entre 2017 e 2025. Em 2017, observamos aproximadamente 183 interrupções por dia. Em 2025, considerando os dados disponíveis de janeiro a maio, essa média chegou a cerca de 303 interrupções diárias. Isso corresponde a um crescimento descritivo de 65,3% no período.

Além do aumento da média, existem dias muito acima do comportamento usual. O maior valor encontrado na série foi de 1.424 interrupções em um único dia.

É importante destacar que este gráfico não demonstra, por si só, uma relação causal nem permite concluir que o crescimento continuará no mesmo ritmo. O que ele evidencia é uma pressão operacional relevante e uma série com eventos extremos. Nesse contexto, antecipar dias de maior risco pode ser útil para reduzir o tempo de resposta e melhorar a alocação de recursos.

**Transição:** Na prática, a previsão é valiosa porque permite agir antes que o pico de ocorrências se concretize.

---

## Slide 4 — Prever picos permite agir antes da falha

**Apresentador:** Giovanni  
**Tempo aproximado:** 1 minuto

**Fala sugerida:**

Grande parte da rede de distribuição do Distrito Federal é aérea e opera em um ambiente sujeito às características do Cerrado, como períodos de chuva intensa, ventos, calor e vegetação próxima à rede.

Hoje, muitas ações ocorrem de forma reativa: a interrupção acontece, o chamado é registrado e, então, a equipe é mobilizada. Uma previsão confiável não elimina a falha, mas cria uma janela para preparação.

Com essa antecedência, a distribuidora pode posicionar equipes em regiões estratégicas, conferir a disponibilidade de materiais, reforçar canais de comunicação e priorizar inspeções ou manutenções em períodos de maior risco.

Portanto, nosso objetivo não é substituir a decisão humana, mas produzir informação antecipada que torne a resposta operacional mais preparada.

**Transição:** A partir dessa motivação, estruturamos o trabalho em quatro contribuições principais.

---

## Slide 5 — O trabalho entrega quatro contribuições

**Apresentador:** Giovanni  
**Tempo aproximado:** 1 minuto e 10 segundos

**Fala sugerida:**

A primeira contribuição foi construir uma base integrada com 3.073 dias de dados brutos, reunindo informações que originalmente estavam separadas em diferentes fontes públicas.

A segunda foi transformar essas informações em 40 atributos temporais, capazes de representar memória recente, tendência, variabilidade e sazonalidade.

A terceira foi realizar uma comparação justa entre os três modelos. Para isso, preservamos a ordem temporal, evitamos vazamento de dados e utilizamos exatamente os mesmos 365 dias de teste em todos os casos.

Por fim, não buscamos apenas um vencedor geral. Avaliamos qual arquitetura se adapta melhor a cada horizonte de previsão, porque uma escolha operacional para o dia seguinte pode ser diferente da escolha para sete ou quatorze dias.

**Transição:** Para viabilizar essa análise, utilizamos três conjuntos de dados públicos.

---

## Slide 6 — Três fontes públicas sustentam o estudo

**Apresentador:** Giovanni  
**Tempo aproximado:** 1 minuto e 20 segundos

**Fala sugerida:**

A base climática veio do INMET, principalmente da estação A001, localizada em Brasília. Ela fornece variáveis como temperatura, precipitação e características do vento.

Os registros de interrupções foram obtidos na ANEEL e agregados por dia para formar nossa variável-alvo. Também utilizamos dados mensais de consumo do SAMP, que acrescentam uma aproximação da carga e do contexto operacional ao longo do tempo.

O período analisado vai de 2017 a 2025. Inicialmente, reunimos 3.073 dias. Depois do alinhamento das datas, da criação dos atributos defasados e do tratamento dos dados, a base final ficou com 3.066 observações diárias utilizáveis.

Nosso recorte é a distribuição de energia no Distrito Federal, em resolução diária. Essa delimitação deve ser lembrada: o resultado caracteriza o comportamento agregado do DF, e não um alimentador ou uma região específica.

**Transição:** Depois da coleta, essas fontes passaram por um único pipeline de processamento.

---

## Slide 7 — O pipeline integra dados e gera previsões

**Apresentador:** Giovanni  
**Tempo aproximado:** 1 minuto e 20 segundos

**Fala sugerida:**

O pipeline começa com as três fontes: INMET, ANEEL e SAMP. Em seguida, todas as informações são convertidas para uma escala diária e alinhadas pela data.

Para variáveis acumulativas, como precipitação e quantidade de interrupções, usamos a soma diária. Para temperatura e vento, utilizamos medidas representativas do dia, como médias. Pequenas lacunas foram tratadas por interpolação linear, mas os valores extremos reais foram preservados, porque justamente esses dias podem carregar informações importantes.

Após a integração, geramos os 40 atributos temporais. Esses atributos alimentam o XGBoost e as duas redes recorrentes. Para cada arquitetura, treinamos modelos diretos e independentes para prever um, três, sete ou quatorze dias à frente.

Assim, a previsão de quatorze dias não depende de previsões intermediárias. Isso evita a propagação recursiva de erro ao longo do horizonte.

**Transição:** A etapa seguinte foi representar, nos atributos, tanto a memória da série quanto sua sazonalidade.

---

## Slide 8 — Atributos codificam memória e sazonalidade

**Apresentador:** Giovanni  
**Tempo aproximado:** 1 minuto e 10 segundos

**Fala sugerida:**

Os atributos foram organizados em dois grandes grupos. O primeiro representa a memória recente da série. Nele, incluímos defasagens de um a sete dias, médias móveis exponenciais de três, sete e quatorze dias e o desvio-padrão móvel de sete dias. Para as redes recorrentes, também utilizamos uma janela de entrada com quatorze dias.

O segundo grupo representa o contexto climático e sazonal. Ele inclui temperatura, precipitação, cinco variáveis relacionadas ao vento, consumo mensal, dia da semana, posição no ano e codificação cíclica do mês com seno e cosseno.

Em todos os horizontes, a informação disponível até o instante atual é usada para prever o valor no instante futuro. Dessa forma, nenhum atributo utiliza dados posteriores à data em que a previsão seria feita.

**Transição:** Esse mesmo cuidado foi aplicado à divisão entre treinamento, validação e teste.

---

## Slide 9 — O teste preserva a ordem do tempo

**Apresentador:** Giovanni  
**Tempo aproximado:** 1 minuto e 30 segundos

**Fala sugerida:**

Em séries temporais, uma divisão aleatória pode fazer o modelo aprender com o futuro e ser avaliado no passado. Por isso, preservamos rigorosamente a ordem cronológica.

O período de treinamento e validação vai de 8 de janeiro de 2017 até 31 de maio de 2024. Dentro desse intervalo, usamos cinco divisões temporais para selecionar os modelos sem embaralhar as observações.

O teste final corresponde ao período de 1º de junho de 2024 a 31 de maio de 2025, totalizando 365 datas-alvo. Essas mesmas datas foram utilizadas por todos os modelos e em todos os horizontes, garantindo comparabilidade.

Outro cuidado foi ajustar a normalização apenas com os dados de treinamento. Se utilizássemos estatísticas do conjunto de teste, mesmo indiretamente, teríamos vazamento de informação.

Por fim, treinamos um modelo independente para cada horizonte. Portanto, cada resultado apresentado adiante corresponde a uma previsão direta e a um protocolo temporal controlado.

**Transição:** Com o protocolo definido, comparamos três estratégias de modelagem.

---

## Slide 10 — Três modelos, três estratégias

**Apresentador:** Giovanni  
**Tempo aproximado:** 1 minuto e 20 segundos

**Fala sugerida:**

O primeiro modelo foi o XGBoost, um conjunto sequencial de árvores de decisão. Ele recebe diretamente os 40 atributos tabulares e corrige, a cada nova árvore, parte dos erros cometidos pelas anteriores. A configuração selecionada utilizou 300 árvores, profundidade máxima igual a quatro e busca de hiperparâmetros com validação temporal.

O segundo modelo foi a Bi-LSTM, que utiliza células com mecanismos de memória e esquecimento. O terceiro foi a Bi-GRU, que segue uma lógica recorrente semelhante, mas com uma estrutura mais compacta.

Para manter a comparação equilibrada, as duas redes receberam sequências de quatorze dias, duas camadas bidirecionais com 64 unidades e dropout de 0,4.

Os modelos foram avaliados por MAE, RMSE, coeficiente de determinação, ou R², e MAPE. Cada métrica destaca um aspecto diferente do erro, por isso analisamos o conjunto delas.

**Transição:** Antes dos resultados dos modelos, vale observar o comportamento da própria série.

---

## Slide 11 — A série revela tendência e sazonalidade

**Apresentador:** Giovanni  
**Tempo aproximado:** 1 minuto e 20 segundos

**Fala sugerida:**

Aqui vemos toda a série diária de interrupções entre 2017 e 2025. Além da tendência de crescimento já apresentada, existe uma sazonalidade visível.

Os maiores valores aparecem com mais frequência entre outubro e março, período que coincide com a estação chuvosa no Distrito Federal. Já entre maio e agosto, os níveis tendem a ser menores. Também observamos picos isolados superiores a mil interrupções, que são raros, mas têm grande impacto operacional e estatístico.

Essa combinação de tendência, sazonalidade e eventos extremos torna o problema desafiador. Um modelo precisa acompanhar o comportamento cotidiano sem ignorar os dias críticos, que são justamente os mais difíceis de prever.

Até aqui, eu apresentei o contexto, os dados e o desenho experimental. Agora o Mateus vai mostrar o que a análise temporal revelou e como os modelos se comportaram.

**Transição e troca:** Giovanni passa a palavra para Mateus.

---

## Slide 12 — A série mantém memória além de 14 dias

**Apresentador:** Mateus  
**Tempo aproximado:** 1 minuto e 15 segundos

**Fala sugerida:**

Obrigado, Giovanni. Começando pela dependência temporal, estes gráficos mostram a autocorrelação e a autocorrelação parcial da série.

A autocorrelação indica que o número de interrupções de um dia mantém relação estatística com valores observados em dias anteriores. Essa dependência não desaparece imediatamente e continua significativa em defasagens superiores a quatorze dias.

Esse resultado sustenta duas decisões do trabalho. A primeira foi criar atributos explícitos de memória, como defasagens e médias móveis exponenciais. A segunda foi fornecer às redes recorrentes uma sequência de quatorze dias, para que elas pudessem aprender padrões de evolução no tempo.

Isso não significa que quatorze dias contenham toda a memória existente. É um compromisso entre contexto temporal, quantidade de dados disponíveis e complexidade do treinamento.

**Transição:** Além da memória da própria série, analisamos como o sinal climático aparece em diferentes escalas.

---

## Slide 13 — A agregação revela melhor o sinal climático

**Apresentador:** Mateus  
**Tempo aproximado:** 1 minuto e 30 segundos

**Fala sugerida:**

Neste gráfico, comparamos correlações de Pearson em diferentes escalas de agregação. Em escala diária, muitas relações são enfraquecidas por ruído, atrasos de efeito e variações locais. Quando agregamos os dados em períodos maiores, parte desse sinal se torna mais visível.

Na escala mensal, por exemplo, encontramos correlação de 0,59 com a direção do vento, 0,54 com a precipitação e 0,48 com o consumo. Esses valores sugerem que clima e contexto operacional carregam informação relevante para a série de interrupções.

Entretanto, é essencial interpretar corretamente esse resultado. Correlação não significa causalidade, e algumas variáveis também compartilham a mesma sazonalidade anual. Portanto, não afirmamos que uma mudança isolada em uma delas cause diretamente uma quantidade específica de interrupções.

O resultado serve como evidência exploratória e como justificativa para incluir essas informações nos modelos preditivos.

**Transição:** Com esses sinais incorporados, começamos comparando o horizonte mais curto, de um dia.

---

## Slide 14 — XGBoost e Bi-LSTM empatam em um dia

**Apresentador:** Mateus  
**Tempo aproximado:** 1 minuto e 30 segundos

**Fala sugerida:**

Para a previsão de um dia à frente, XGBoost e Bi-LSTM apresentaram resultados tecnicamente próximos.

O XGBoost obteve o menor MAE, de 61,09 interrupções, e o menor MAPE, de 20,64%. A Bi-LSTM apresentou o menor RMSE, de 98,80, e o maior R², de 0,420. A diferença de MAE entre os dois modelos foi de apenas 2,7%.

Já a Bi-GRU teve desempenho inferior nesse horizonte, com MAE de 73,44 e R² de 0,156.

Portanto, não seria correto afirmar que existe um vencedor absoluto para um dia. O XGBoost é ligeiramente melhor no erro absoluto médio e é mais simples operacionalmente. A Bi-LSTM, por outro lado, reduz um pouco mais os erros quadráticos e explica uma parcela ligeiramente maior da variância.

A escolha entre os dois depende da métrica priorizada e do custo de implantação.

**Transição:** A média, porém, não conta toda a história; precisamos observar onde os erros se concentram.

---

## Slide 15 — Os modelos suavizam os maiores picos

**Apresentador:** Mateus  
**Tempo aproximado:** 1 minuto e 10 segundos

**Fala sugerida:**

Este gráfico compara os valores reais com as previsões do XGBoost no conjunto de teste para um dia à frente.

O modelo acompanha razoavelmente a faixa mais frequente da série e reconhece parte das variações ao longo do período. Entretanto, nos maiores picos, a curva prevista tende a ficar abaixo da curva real.

Esse comportamento é comum quando há poucos eventos extremos no treinamento. Para minimizar o erro médio, o modelo aprende previsões mais próximas da região central da distribuição e acaba suavizando ocorrências raras.

Assim, o modelo pode ser útil para antecipar o nível geral de demanda operacional, mas ainda não reproduz com precisão os dias de maior criticidade. Essa limitação aparece de forma ainda mais clara quando separamos os erros por severidade.

**Transição:** Quando classificamos os dias por quantidade de interrupções, o erro cresce fortemente na categoria mais crítica.

---

## Slide 16 — O erro cresce nos dias mais críticos

**Apresentador:** Mateus  
**Tempo aproximado:** 1 minuto e 20 segundos

**Fala sugerida:**

Aqui dividimos os dias de teste em três faixas de severidade. Nos dias considerados normais, com menos de 200 interrupções, o MAE do XGBoost foi de 30,1. Nos dias moderados, entre 200 e 400, o erro subiu para 38,9.

Já nos dias severos, com mais de 400 interrupções, o MAE chegou a 135,9. Esses dias representam apenas 13,7% do conjunto, mas são justamente os mais relevantes para uma operação de contingência.

O resultado mostra uma heteroscedasticidade: a magnitude e a dispersão do erro aumentam com a severidade do evento.

Por isso, uma implantação real não deveria comunicar somente uma previsão pontual, como “esperamos 350 interrupções”. O próximo passo mais seguro seria fornecer também intervalos de previsão ou probabilidades de ultrapassar níveis críticos.

**Transição:** Para entender o bom desempenho de curto prazo, analisamos quais atributos o XGBoost mais utilizou.

---

## Slide 17 — O histórico domina o curto prazo

**Apresentador:** Mateus  
**Tempo aproximado:** 1 minuto e 15 segundos

**Fala sugerida:**

Neste gráfico estão os atributos mais importantes para o XGBoost na previsão de um dia à frente.

O número de interrupções mais recente e a primeira defasagem aparecem entre as variáveis de maior importância. Isso confirma que o estado atual da rede contém um forte sinal para o dia seguinte.

Também aparecem atributos de sazonalidade, como o mês, e variáveis climáticas acumuladas, como a média móvel exponencial da precipitação. Portanto, o modelo combina memória imediata com contexto climático e posição no calendário.

Novamente, essa importância deve ser interpretada como contribuição preditiva, e não como prova de causalidade. Um atributo pode ser importante porque resume condições compartilhadas com outras variáveis.

O resultado ajuda a explicar por que um modelo de árvores, alimentado por bons atributos temporais, consegue competir com redes recorrentes no horizonte de um dia.

**Transição:** Quando ampliamos o horizonte, entretanto, a hierarquia entre os modelos muda.

---

## Slide 18 — A hierarquia muda com o horizonte

**Apresentador:** Mateus  
**Tempo aproximado:** 1 minuto e 40 segundos

**Fala sugerida:**

Este é o principal resultado comparativo do trabalho. O gráfico mostra o menor MAE encontrado em cada horizonte.

Para um dia à frente, o XGBoost obteve MAE de 61,1. Em três dias, o melhor resultado foi da Bi-LSTM, com 72,1. Em sete dias, a Bi-GRU alcançou 71,8 e, em quatorze dias, também foi a melhor, com MAE de 62,8.

Todos esses valores foram calculados sobre as mesmas 365 datas-alvo, e cada horizonte utilizou um modelo direto e independente.

É importante não interpretar a queda do MAE em quatorze dias como se prever duas semanas fosse necessariamente mais fácil. A distribuição das datas-alvo muda com o alinhamento temporal, e cada modelo é treinado separadamente. O ponto central é a comparação entre arquiteturas dentro de cada horizonte.

Os resultados indicam que os atributos tabulares são muito eficientes no curto prazo, enquanto a capacidade das redes recorrentes de representar sequências se torna mais vantajosa nos horizontes maiores.

**Transição:** Isso nos leva a uma recomendação baseada no horizonte, e não em um único modelo universal.

---

## Slide 19 — Não existe um único modelo vencedor

**Apresentador:** Mateus  
**Tempo aproximado:** 1 minuto e 20 segundos

**Fala sugerida:**

Em vez de escolher uma única arquitetura para todos os casos, propomos uma combinação guiada pela necessidade operacional.

Para um dia à frente, XGBoost ou Bi-LSTM podem apoiar alertas de curto prazo. O XGBoost tem a vantagem de menor custo computacional e maior facilidade de interpretação.

Para três dias, a Bi-LSTM apresentou o melhor MAE e pode apoiar decisões sobre equipes e materiais. Para sete e quatorze dias, a Bi-GRU teve o melhor desempenho e pode contribuir para o planejamento logístico de médio prazo.

Esse arranjo não precisa ser estático. Em produção, os modelos devem ser monitorados e retreinados periodicamente, porque clima, consumo, infraestrutura e processos operacionais mudam com o tempo.

Portanto, a principal recomendação é manter modelos especializados por horizonte e escolher a saída de acordo com a decisão que precisa ser tomada.

**Transição:** Antes de propor uma implantação, precisamos deixar claras as limitações do estudo.

---

## Slide 20 — O escopo dos resultados ainda é restrito

**Apresentador:** Mateus  
**Tempo aproximado:** 1 minuto e 20 segundos

**Fala sugerida:**

O trabalho possui limitações de dados e de modelagem. No aspecto climático, utilizamos principalmente uma estação meteorológica para representar todo o Distrito Federal. Além disso, a resolução é diária e geograficamente agregada.

Não tivemos acesso a informações detalhadas sobre topologia da rede, idade dos equipamentos, podas, vegetação próxima aos circuitos ou histórico de manutenção. Esses fatores poderiam explicar melhor as diferenças locais.

Nas redes recorrentes, a quantidade de sequências de treinamento foi de aproximadamente 2.687, o que é relativamente pequeno para modelos profundos. A direção do vento também foi agregada por média simples, embora seja uma variável circular. Além disso, produzimos apenas previsões pontuais.

Como continuidade, sugerimos dados horários e espaciais, múltiplas estações, informações de vegetação como NDVI ou LiDAR, indicadores climáticos como ENSO e modelos probabilísticos.

**Transição:** Mesmo com essas limitações, a solução já permite desenhar um primeiro sistema de apoio à decisão.

---

## Slide 21 — A implantação pode começar como alerta

**Apresentador:** Mateus  
**Tempo aproximado:** 1 minuto

**Fala sugerida:**

Uma implantação inicial pode funcionar como um sistema de alerta, sem automatizar decisões críticas.

As bases seriam atualizadas diariamente e processadas pelo mesmo pipeline. Em seguida, cada horizonte utilizaria o modelo com melhor desempenho para gerar um nível de risco.

Esse risco poderia ser comparado a limites definidos pela operação. Dependendo do resultado, a distribuidora poderia reforçar equipes, reservar materiais, ajustar a comunicação com consumidores ou priorizar inspeções.

O sistema também precisaria acompanhar continuamente o erro, mudanças na distribuição dos dados e sinais de degradação do modelo. Quando necessário, os modelos seriam retreinados.

Assim, a previsão funciona como apoio à decisão humana: ela organiza evidências e antecipa risco, mas a ação final continua sob responsabilidade da equipe técnica.

**Transição:** Com isso, chegamos às conclusões do trabalho.

---

## Slide 22 — A melhor arquitetura depende do horizonte

**Apresentador:** Mateus  
**Tempo aproximado:** 1 minuto e 20 segundos

**Fala sugerida:**

Retomando nossa pergunta inicial, os resultados indicam que é possível prever interrupções diárias no Distrito Federal usando dados públicos, com utilidade potencial para o planejamento operacional.

Quatro conclusões principais emergem do estudo. Primeiro, o histórico da própria série, o clima e a sazonalidade contêm sinal preditivo relevante. Segundo, para um dia à frente, XGBoost e Bi-LSTM apresentam desempenho próximo, com vantagens em métricas diferentes.

Terceiro, em horizontes maiores, as redes recorrentes passam a apresentar vantagem: a Bi-LSTM em três dias e a Bi-GRU em sete e quatorze dias. Quarto, os eventos extremos continuam sendo o maior desafio, pois todos os modelos tendem a suavizar os picos.

Portanto, a melhor solução não é uma arquitetura única. É uma estratégia que selecione o modelo conforme o horizonte e que, futuramente, comunique também a incerteza associada à previsão.

**Transição:** Encerramos aqui a apresentação e ficamos à disposição para as perguntas.

---

## Slide 23 — Obrigado! Perguntas e discussão

**Apresentadores:** Giovanni e Mateus  
**Tempo aproximado:** 15 segundos, antes das perguntas

**Fala sugerida — Giovanni:**

Obrigado pela atenção. Agradecemos também ao nosso orientador e à banca. Estamos à disposição para perguntas e comentários.

**Condução sugerida durante a arguição:**

- Giovanni responde primeiro a perguntas sobre motivação, fontes de dados, construção da base, atributos, divisão temporal e configuração dos modelos.
- Mateus responde primeiro a perguntas sobre métricas, interpretação dos gráficos, comparação por horizonte, limitações, conclusões e implantação.
- Se a pergunta envolver os dois blocos, quem começar deve concluir a ideia principal e depois passar a palavra, por exemplo: “O Mateus pode complementar com o impacto desse ponto nos resultados”.
- Evitem responder simultaneamente. Façam uma pausa curta antes de começar e confirmem se a pergunta foi totalmente respondida.

---

# Slides de reserva

Os próximos slides não fazem parte dos 28 minutos da apresentação principal. Abram apenas quando a pergunta da banca estiver diretamente relacionada ao conteúdo.

## Slide 24 — Métricas completas confirmam a inversão

**Apresentador principal:** Mateus  
**Quando usar:** pergunta sobre métricas completas, comparação justa ou desempenho em cada horizonte.

**Resposta sugerida:**

Esta tabela apresenta as métricas completas para todos os modelos e horizontes. Nossa conclusão não foi baseada somente no MAE: também observamos RMSE e R². No horizonte de um dia, XGBoost e Bi-LSTM permanecem muito próximos. Nos horizontes maiores, as redes recorrentes passam a apresentar resultados mais favoráveis, especialmente a Bi-GRU em sete e quatorze dias. Um exemplo é o horizonte de quatorze dias, no qual a Bi-GRU alcançou R² de 0,351. Todos os valores utilizam as mesmas 365 datas-alvo dentro de cada comparação.

---

## Slide 25 — XGBoost tem menor custo operacional

**Apresentador principal:** Giovanni  
**Quando usar:** pergunta sobre custo computacional, implantação ou escolha prática entre XGBoost e redes neurais.

**Resposta sugerida:**

Em nossos experimentos, o XGBoost apresentou o menor custo de treinamento e não exigiu GPU. O treinamento ficou abaixo de dois minutos em CPU, enquanto as redes levaram aproximadamente vinte a vinte e cinco minutos com aceleração por GPU. Na inferência diária, todos os modelos são suficientemente rápidos para o problema, com tempos da ordem de milissegundos. Assim, a principal diferença operacional aparece no ciclo de desenvolvimento, ajuste e retreinamento. Os valores desta tabela são estimativas do nosso ambiente experimental e podem mudar conforme o hardware e a implementação.

---

## Slide 26 — As redes concentram erros perto de zero

**Apresentador principal:** Mateus  
**Quando usar:** pergunta sobre distribuição dos resíduos ou aparente divergência entre métricas.

**Resposta sugerida:**

Estas curvas mostram a distribuição dos resíduos. A Bi-LSTM e a Bi-GRU apresentam maior concentração de erros perto de zero, enquanto o XGBoost produz uma distribuição mais espalhada e conservadora. Isso não contradiz o resultado de MAE, porque uma distribuição pode ter muitos erros pequenos e, ao mesmo tempo, algumas caudas mais intensas. As métricas resumem a distribuição de maneiras diferentes. Por esse motivo, analisamos MAE, RMSE, R², MAPE, comportamento temporal e severidade, em vez de decidir com base em um único número.

---

## Slide 27 — O trabalho é integralmente reprodutível

**Apresentador principal:** Giovanni  
**Quando usar:** pergunta sobre código, dados, repetição do experimento ou referências das arquiteturas.

**Resposta sugerida:**

O código do pipeline, os procedimentos de tratamento e os experimentos estão organizados no repositório indicado no slide. Os dados utilizados são públicos e vêm do INMET, da ANEEL e do SAMP. Também documentamos as referências centrais das arquiteturas, incluindo os trabalhos do XGBoost, da LSTM e da GRU. Com os mesmos dados, parâmetros, sementes e ambiente de software, o experimento pode ser reproduzido. Pequenas diferenças numéricas ainda podem ocorrer por versão de bibliotecas, hardware e operações não determinísticas das redes neurais.

---

# Recomendações para o ensaio

1. Façam pelo menos um ensaio individual e dois ensaios juntos com cronômetro.
2. Busquem concluir o slide 11 por volta de 13 minutos e o slide 22 entre 27 e 28 minutos.
3. Se estiverem atrasados, reduzam exemplos e transições; não acelerem a leitura dos números.
4. Nos slides com tabelas, destaquem somente os valores necessários para a conclusão.
5. No slide 18, enfatizem que a comparação é feita dentro de cada horizonte.
6. Usem os slides de reserva apenas durante a arguição, evitando prolongar a exposição principal.
7. Ao receber uma pergunta, repitam brevemente o ponto central antes de responder. Isso ajuda a organizar a resposta e garante que toda a banca tenha ouvido a questão.
