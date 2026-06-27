#%% 1. Importação de pacotes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Pacotes para modelagem 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

#%% 2. Carregamento e Pré-processamento 
# Carregando a base de dados enviada
df = pd.read_csv(r'C:\Users\pc\Desktop\TCC USP\Python_M11_support material .csv', delimiter=',')

def tratar_moeda(coluna):
    return pd.to_numeric(coluna.str.replace('.', '', regex=False).str.replace(',', '.', regex=False))

df['limite_credito'] = tratar_moeda(df['limite_credito'])
df['valor_transacoes_12m'] = tratar_moeda(df['valor_transacoes_12m'])

# Variável alvo: 1 para Inadimplente (Classe de interesse), 0 para Bom Pagador
df['inadimplente'] = df['default']

# Transformação em Dummies (drop_first=False para melhor leitura das regras da árvore)
df_final = pd.get_dummies(df, columns=['sexo', 'escolaridade', 'estado_civil', 'salario_anual'], drop_first=False)

#%% 3. Definição de Variáveis e Divisão Treino/Teste 
# Selecionando as variáveis preditoras (X) e a resposta (y)
X = df_final.drop(columns=['id', 'default', 'inadimplente', 'tipo_cartao'])
y = df_final['inadimplente']

# Divisão em 70% treino e 30% teste com estratificação para manter a proporção de classes [10, 11]
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

#%% 4. Treinamento da Random Forest 
# Configuração baseada na teoria de Breiman: mtry = sqrt(p) e sem poda
modelo_floresta = RandomForestClassifier(
    n_estimators=500,      # Número de árvores (estabilidade estatística)
    max_features='sqrt',   # Parâmetro 'mtry': raiz quadrada das variáveis preditoras
    bootstrap=True,        # Seleção de amostras com reposição (Bagging) 
    oob_score=True,        # Cálculo do erro Out-of-Bag (validação interna gratuita) 
    random_state=42
)

modelo_floresta.fit(X_treino, y_treino)

#%% 5. Análise de Importância das Variáveis
# Identifica quais atributos mais contribuíram para a predição
importancias = pd.Series(modelo_floresta.feature_importances_, index=X.columns)
plt.figure(figsize=(10,6))
importancias.sort_values(ascending=False).head(10).plot(kind='bar', color='skyblue')
plt.title("Figura X: Top 10 Variáveis mais importantes (Random Forest)")
plt.ylabel("Ganho de Pureza (Gini)")
plt.show()

#%% 6. Avaliação de Performance no Conjunto de TESTE 
# Predição em dados que o modelo nunca viu
y_pred = modelo_floresta.predict(X_teste)

# Matriz de Confusão para análise de erros (Falsos Negativos) 
plt.figure(figsize=(8,6))
ConfusionMatrixDisplay.from_predictions(
    y_teste, y_pred, 
    display_labels=['Bom Pagador', 'Inadimplente'],
    cmap='Blues'
)
plt.title("Matriz de Confusão: Random Forest")
plt.show()

# Relatório detalhado com foco no Recall (Sensibilidade) da classe 1
print("--- Relatório de Classificação (Dados de Teste) ---")
print(classification_report(y_teste, y_pred))

# Exibindo o erro OOB (Out-of-Bag) 
print(f"Acurácia Out-of-Bag (OOB Score): {modelo_floresta.oob_score_:.4f}")
