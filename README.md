# ml_cartola

**Descrição**

Nesta tarefa construiremos modelos preditivos de regressão para a predição de notas de jogadores no Cartola (2014). Os dados a serem utilizados estão disponíveis  [aqui](https://canvas.instructure.com/courses/1668248/files/82677186/download?wrap=1 "cartola_2014.csv")[![Visualizar o documento](https://canvas.instructure.com/images/preview.png)](https://canvas.instructure.com/courses/1668248/files/82677186/download?wrap=1 "Visualizar o documento")  e seguem a seguinte descrição:

atleta_id: Referência ao atleta  
rodada_id: Rodada atual  
clube_id: Referência ao clube  
posicao_id: Referência à posição  
status_id: Referência ao status  
pontos_num: Pontos obtidos na rodada atual  
preco_num: Preço atual  
variacao_num: Variação de Preço desde a última rodada  
media_num: Média de pontos por rodada jogada  
jogos_num: Número de partidas jogadas  
FS: Faltas sofridas  
PE: Passes perdidos  
A: Assistências  
FT: Chutes na trave  
FD: Chutes defendidos  
FF: Chutes  
G: Gols  
I: Impedimentos  
PP: Penaltis perdidos  
RB: successful tackes  
FC: Faltas cometidas  
GC: Gols contra  
CA: Cartões amarelos  
CV: Cartões vermelhos  
SG: Sem gols (apenas defesa)  
DD: Defesas difíceis (somente goleiros)  
DP: Penaltis defendidos (somente goleiros)  
GS: Gols sofridos (somente goleiros)  
**Nota**: variável alvo

As atividades esperadas para essa etapa são descritas a seguir:

1.  Realize uma análise exploratória nos dados, identificando e explorando: (i) correlações entre as variáveis, (ii) distribuição das variáveis e (iii) valores ausentes (10 pts.)
2.  Usando todas as variáveis disponíveis, tune (usando validação cruzada): (i) um modelo de regressão Ridge, (ii) um modelo de regressão Lasso e (iii) um modelo KNN. Para os modelos de regressão linear, o parâmetro a ser tunado é o lambda (penalização dos coeficientes) e no KNN o número (K) de vizinhos. Compare os três modelos em termos do erro RMSE de validação cruzada. (30 pts.)
3.  Quais as variáveis mais importantes segundo o modelo de regressão Ridge e Lasso? Variáveis foram descartadas pelo Lasso? Quais? (10 pts.)
4.  Re-treine o melhor modelo (usando os melhores valores de parâmetros encontrados em todos os dados, sem usar validação cruzada). Use esse último modelo treinado para prever os dados de teste disponíveis no challenge que criamos na plataforma Kaggle (30 pts.)

**Ideias para melhorar os resultados:**

-   Tente criar novas variáveis a partir das variáveis existentes.
-   Tente usar transformações (e.g. log) nas variáveis que apresentarem viés.
-   Tente normalizar as variáveis.
-   Experimente estratégias diferentes para lidar com dados ausentes (NAs).
-   Use outros métodos de regressão. Por exemplo, SVR, Árvores de Regressão e Florestas Aleatórias.

**Como devo entregar?**

Link para o Colab. Além disso, pelo menos uma submissão deve ser feita no desafio que lançaremos no Kaggle.

**Link para competição no Kaggle:** [CartolaFC](https://www.kaggle.com/c/ufcg-cdp-20192/overview)