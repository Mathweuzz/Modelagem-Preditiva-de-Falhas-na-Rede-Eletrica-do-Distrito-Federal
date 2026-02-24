# Segundo Pedido – Planejamento da Nova Entrega

Pasta da entrega: `SegundoPedido/`  
Arquivo de planejamento: `SegundoPedido/SegundoPedido.md`

---

## 1. Séries semanais e mensais: interrupções x temperatura

1.1. Gerar séries **semanais** de:
- Número de interrupções por dia/semana
- Temperatura média (e/ou máxima/mínima) na semana

1.2. Gerar séries **mensais** de:
- Número de interrupções por mês
- Temperatura média (e/ou máxima/mínima) no mês

1.3. Produzir gráficos:
- Interrupções x tempo (semanal e mensal)
- Temperatura x tempo (semanal e mensal)
- Se fizer sentido, gráficos combinados (ex.: dois eixos: interrupções e temperatura)

1.4. Salvar saídas em:
- Dados agregados: `SegundoPedido/dados/aggregados_semanal_mensal_interrupcoes_tempo.csv`
- Gráficos: `SegundoPedido/graficos/` (nomes tipo `interrupcoes_vs_temperatura_semanal.png`, `..._mensal.png`)

---

## 2. Séries semanais e mensais: interrupções x precipitação

2.1. Gerar séries **semanais** de:
- Número de interrupções por semana
- Precipitação total ou média na semana

2.2. Gerar séries **mensais** de:
- Número de interrupções por mês
- Precipitação total ou média no mês

2.3. Produzir gráficos:
- Interrupções x tempo (semanal e mensal)
- Precipitação x tempo (semanal e mensal)
- Gráficos combinados, se for útil, para visualizar correlação visual

2.4. Salvar saídas em:
- Dados agregados: `SegundoPedido/dados/aggregados_semanal_mensal_interrupcoes_precipitacao.csv`
- Gráficos: `SegundoPedido/graficos/` (ex.: `interrupcoes_vs_precipitacao_semanal.png`, `..._mensal.png`)

---

## 3. Médias móveis nas séries diárias

3.1. Definir janelas de média móvel (por exemplo, 7 dias, 14 dias):
- Para número diário de interrupções
- Opcional: também para temperatura e precipitação diárias

3.2. Gerar séries com média móvel:
- Interrupções diárias + curva de média móvel
- (Opcional) Temperatura diária + média móvel
- (Opcional) Precipitação diária + média móvel

3.3. Produzir gráficos:
- Gráficos diários com a série original e a média móvel sobreposta

3.4. Salvar saídas em:
- Dados com colunas de média móvel: `SegundoPedido/dados/serie_diaria_com_mm.csv`
- Gráficos: `SegundoPedido/graficos/` (ex.: `diario_interrupcoes_media_movel.png`)

---

## 4. Revisão do procedimento de previsão (treino x teste)

4.1. Explicitar claramente a regra de separação temporal:
- Exemplo sugerido pelo professor: usar **todos os dias até 02/11** para prever **03/11** (e assim por diante)
- Garantir que **nenhum dia usado como alvo de previsão** esteja dentro do conjunto de treino

4.2. Verificar o que foi feito na primeira versão:
- Confirmar se o conjunto de treinamento e teste respeitou a ordem temporal
- Caso não tenha respeitado, **refazer o treinamento** com divisão correta

4.3. Re-treinar os modelos de previsão (se necessário):
- Aplicar a divisão temporal correta (treino até certo dia, previsão a partir do dia seguinte)
- Documentar quantos dias foram usados para treino, quantos para teste

4.4. Avaliar e registrar métricas:
- RMSE, MAE, MAPE ou outras métricas escolhidas
- Tabelas e/ou gráficos comparando valores previstos x observados

4.5. Salvar saídas em:
- Scripts ou notebooks: `SegundoPedido/modelos/`
- Resultados (tabelas, métricas): `SegundoPedido/resultados/`
- Gráficos de previsão: `SegundoPedido/graficos/previsao_*.png`

---

## 5. Integração dos dados de consumo diário

5.1. Obter e preparar a base de **consumo diário**:
- Carregar dados de consumo para o mesmo período das interrupções
- Padronizar datas, unidades e formatos

5.2. Integrar consumo com interrupções:
- Montar tabela diária com:
  - Data
  - Número de interrupções
  - Consumo total do dia
  - Temperatura
  - Precipitação

5.3. Salvar base integrada:
- `SegundoPedido/dados/base_diaria_integrada.csv`

---

## 6. Gráficos: falhas (interrupções) x consumo

6.1. Gráficos diários:
- Interrupções por dia x consumo diário (ex.: dois eixos ou gráficos separados)
- Possíveis scatter plots: (consumo diário, número de interrupções)

6.2. Gráficos semanais e mensais (opcional mas desejável):
- Agregar consumo por semana/mês
- Repetir os gráficos de interrupções x consumo em base semanal/mensal

6.3. Análises simples:
- Verificar visualmente se dias de maior consumo parecem ter mais interrupções
- Calcular correlações simples (ex.: coeficiente de correlação de Pearson)

6.4. Salvar saídas em:
- Dados agregados: `SegundoPedido/dados/interrupcoes_consumo_aggregados.csv`
- Gráficos: `SegundoPedido/graficos/interrupcoes_vs_consumo_*.png`

---

## 7. Gráficos: consumo x temperatura

7.1. Gráficos diários:
- Consumo diário x temperatura diária (ex.: scatter e/ou séries temporais sobrepostas)

7.2. Gráficos semanais e mensais:
- Consumo médio/total na semana/mês x temperatura média na semana/mês

7.3. Análises simples:
- Verificar visualmente se consumo aumenta em dias mais quentes ou frios
- Calcular correlações entre consumo e temperatura

7.4. Salvar saídas em:
- Dados agregados: `SegundoPedido/dados/consumo_temperatura_aggregados.csv`
- Gráficos: `SegundoPedido/graficos/consumo_vs_temperatura_*.png`

---

## 8. Texto para o relatório / TCC (Segunda Entrega)

8.1. Redigir seção de **Metodologia adicional**:
- Explicar agregações semanais/mensais
- Explicar uso de média móvel nas séries diárias
- Detalhar a nova regra de separação treino/teste

8.2. Redigir seção de **Resultados adicionais**:
- Descrever os padrões observados nos gráficos semanais/mensais
- Comentar o efeito da média móvel na interpretação das séries diárias
- Apresentar e discutir:
  - Relação interrupções x temperatura
  - Relação interrupções x precipitação
  - Relação interrupções x consumo
  - Relação consumo x temperatura

8.3. Redigir um pequeno **subtópico de discussão**:
- Interpretar se os novos gráficos revelam padrões mais claros
- Discutir implicações para previsão de interrupções ou planejamento do sistema elétrico

8.4. Salvar textos em:
- `SegundoPedido/texto/metodologia_segundo_pedido.md`
- `SegundoPedido/texto/resultados_segundo_pedido.md`

---

## 9. Organização geral da pasta SegundoPedido

- `SegundoPedido/dados/` – bases intermediárias e agregadas
- `SegundoPedido/graficos/` – todos os gráficos gerados nesta nova entrega
- `SegundoPedido/modelos/` – scripts ou notebooks de previsão (treino/teste)
- `SegundoPedido/resultados/` – tabelas de métricas, resumos
- `SegundoPedido/texto/` – partes do texto do relatório/TCC
- `SegundoPedido/SegundoPedido.md` – este arquivo de planejamento
