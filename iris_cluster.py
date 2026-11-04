#Abrir os dados
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn import preprocessing
import math
from pickle import dump

dados = pd.read_csv('iris.csv', sep=';')

#1. Balancear os dados (aula 21/10)
from collections import Counter
from imblearn.over_sampling import SMOTE
print('Antes do balanceamento:', Counter(dados['class']))
resampler = SMOTE()
dados_atributos = dados.drop(columns=['class'])  #somente os atributos
dados_classes = dados['class']                   #somente a classe
dados_atributos_b, dados_classes_b = resampler.fit_resample(dados_atributos, dados_classes)
dados = dados_atributos_b.join(dados_classes_b)
print('Depois do balanceamento:', Counter(dados['class']))

#2. Normalizar os dados (aula 28/10)
#Separar as colunas categóricas das numéricas
dados_num = dados.drop(columns=['class'])
dados_cat = dados['class']
dados_cat_rotulos = 'class'

#Instanciar o normalizador
normalizador = preprocessing.MinMaxScaler()
#Treinar o modelo normalizador a partir dos dados numéricos
modelo_normalizador = normalizador.fit(dados_num)
#Salvar o modelo normalizador para uso posterior (normalizar a nova flor e desnormalizar o centróide)
dump(modelo_normalizador, open('modelo_normalizador.model', 'wb'))

#Normalizar os dados numéricos
dados_num_normalizados = modelo_normalizador.transform(dados_num)
#recompor o dataframe com os rótulos de colunas originais
dados_num_normalizados = pd.DataFrame(dados_num_normalizados, columns=dados_num.columns)

#Normalizar os dados categóricos
dados_cat_normalizado = pd.get_dummies(dados_cat, prefix=dados_cat_rotulos, dummy_na=False, dtype=int)

#Juntar em um dataframe único com join
dados_norm = dados_num_normalizados.join(dados_cat_normalizado)
print(dados_norm.head())

#3. Descobrir o número ideal de clusters (aula 21/10)
distorcoes = []
K = range(1, 100)
for i in K:
    cluster_model = KMeans(n_clusters=i, random_state=42).fit(dados_norm)
    distorcoes.append(
        sum(
            np.min(
                cdist(dados_norm, cluster_model.cluster_centers_, 'euclidean'),
                axis=1) / dados_norm.shape[0]
        )
    )

#Método do cotovelo (elbow): ponto mais distante da reta entre o primeiro e o último ponto
x0 = K[0]
y0 = distorcoes[0]
xn = K[-1]
yn = distorcoes[-1]
distancias = []
for i in range(len(distorcoes)):
    x = K[i]
    y = distorcoes[i]
    numerador = abs((yn - y0) * x - (xn - x0) * y + xn * y0 - yn * x0)
    denominador = math.sqrt((yn - y0) ** 2 + (xn - x0) ** 2)
    distancias.append(numerador / denominador)

n_clusters_otimo = K[distancias.index(max(distancias))]
print('Número ótimo de clusters:', n_clusters_otimo)

#4. Treinamento definitivo e salvar o modelo
iris_cluster = KMeans(n_clusters=n_clusters_otimo, random_state=42).fit(dados_norm)
dump(iris_cluster, open('iris_cluster.model', 'wb'))
print('Modelo salvo em iris_cluster.model')
