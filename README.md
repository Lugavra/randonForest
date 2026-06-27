# Classificação de Inadimplência com Random Forest
Este repositório contém o desenvolvimento de um modelo de Machine Learning baseado no algoritmo Random Forest (Floresta Aleatória) para a predição de inadimplência de clientes.O objetivo principal é identificar potenciais inadimplentes com foco na minimização de Falsos Negativos (clientes inadimplentes classificados como bons pagadores), utilizando métricas robustas como Recall (Sensibilidade) e o erro Out-of-Bag (OOB).
🛠️ Tecnologias e Bibliotecas Utilizadas
Python 3.
xPandas & NumPy: Manipulação e pré-processamento de dados.
Matplotlib & Seaborn: Visualização de dados e gráficos de performance.
Scikit-Learn: Divisão metodológica, treinamento do modelo (RandomForestClassifier) e extração de métricas de avaliação.
📊 Estrutura do Pipeline de Código
O script está dividido em 6 etapas lógicas e sequenciais:
1. Importação de Pacotes
   Carga das bibliotecas essenciais para análise de dados, modelagem preditiva e avaliação estatística.
2. Carregamento e Pré-processamento
   Tratamento de Moeda: Conversão de strings de formato monetário brasileiro para o padrão numérico flutuante (float).
   Engenharia de Variáveis: Criação da variável alvo explicativa (inadimplente).
   Codificação de Variáveis Categóricas: Transformação em variáveis dummies via One-Hot Encoding (get_dummies), mantendo todas as colunas (drop_first=False) para        facilitar a interpretação das regras de decisão das árvores.
3. Definição de Variáveis e Divisão Treino/Teste
   Separação dos atributos preditores X e do vetor de resposta Y.
   Divisão metodológica de 70% para treinamento e 30% para teste.
   Utilização de estratificação (stratify=y) para garantir que a proporção de inadimplentes seja idêntica em ambos os conjuntos, mitigando o viés de classes          desbalanceadas.
4. Treinamento da Random Forest
   A configuração do classificador seguiu rigorosamente a fundamentação teórica de Leo Breiman:
   n_estimators=500: Quantidade de árvores para garantir a estabilidade estatística das estimativas.
   max_features='sqrt': Definição do parâmetro mtry (raiz quadrada do total de variáveis preditoras).
   bootstrap=True: Amostragem com reposição (Bagging).
   oob_score=True: Ativação da validação interna por meio do erro Out-of-Bag, permitindo uma avaliação robusta sem a necessidade de gastar dados adicionais de        validação.
5. Análise de Importância das Variáveis
   Geração de um gráfico de barras com as Top 10 variáveis mais impactantes para o modelo, mensuradas pelo critério de ganho de pureza de Gini (Feature              Importance).
6. Avaliação de Performance
   Validação do modelo utilizando o conjunto de dados de teste (dados inéditos):
   Matriz de Confusão: Foco visual no diagnóstico de Falsos Negativos.
   Classification Report: Avaliação de Precision, Recall e F1-Score.
   Acurácia OOB: Score gerado pelas amostras deixadas de fora durante o bootstrap.
