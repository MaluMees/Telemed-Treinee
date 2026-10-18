import pandas as pd
import numpy as np

#Carregando o dataset pelo panda:
data = pd.read_csv('train.csv')

#Tirando colunas específicas:
data.drop(columns=['Utilities', 'Street', 'Condition2', 'RoofMatl', 'Heating', 'PoolQC', 'Exterior1st', 'SaleType'], inplace=True)

#Separando os inputs do alvo a ser previsto ("targets"):
inputs, targets = data.iloc[:, 0:-1], data.iloc[:, -1] #[:, 0:-1] == [linhas, colunas], ":" sozinho seleciona todas as linhas da tabela, "-1" sozinho seleciona exclusivamente a última coluna da tabela(o alvo q sera previsto)
#"0:-1" ta selecionando a primeira até a última coluna

#Tratando valores nulos, selecionando só as colunas numericas e depois fazendo a media dos valores de cada um para substituir em seus respectivos NAs:
colunas_numericas = inputs.select_dtypes(include=['int64', 'float64']).columns
colunas_categoricas = inputs.select_dtypes(include=['object']).columns
inputs[colunas_numericas] = inputs[colunas_numericas].fillna(inputs[colunas_numericas].mean())
inputs[colunas_categoricas] = inputs[colunas_categoricas].fillna('None')

#Aplicando a formula do z-score: (valor original - media) / desvio padrão:
inputs[colunas_numericas] = (inputs[colunas_numericas] - inputs[colunas_numericas].mean()) / inputs[colunas_numericas].std()

#Traduzindo dados ordinais:
legenda_colunas_ordinais = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
colunas_ordinais = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 
                    'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']

for col in colunas_ordinais:
    if col in inputs.columns:
        inputs[col] = inputs[col].map(legenda_colunas_ordinais)

#Calculando one-hot encoding pro resto das categorias:
inputs = pd.get_dummies(inputs)

#Tratando a variavel SalePrice (ln e depois z-score):
log = np.log(targets)
targets = (log - log.mean()) / log.std()
