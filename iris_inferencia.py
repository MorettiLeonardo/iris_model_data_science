#Receber uma nova flor e determinar a qual cluster ela pertence
import pandas as pd
from pickle import load

#carregar o modelo de cluster e o modelo normalizador salvos pelo iris_cluster.py
iris_cluster = load(open('iris_cluster.model', 'rb'))
modelo_normalizador = load(open('modelo_normalizador.model', 'rb'))

#5. Nova instância de flor (pode ser qualquer uma, inclusive da própria base)
nova_flor = pd.DataFrame(
    [[6.1, 2.8, 4.7, 1.2, 'Iris-versicolor']],
    columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
)

#6. Normalizar a nova instância (aula 28/10)
#numéricos: usar o modelo normalizador salvo e o método transform() (nunca fit de novo)
nova_num = nova_flor.drop(columns=['class'])
nova_num_normalizada = pd.DataFrame(modelo_normalizador.transform(nova_num), columns=nova_num.columns)

#categóricos: get_dummies e reindex para ter as mesmas colunas do treino
#(uma flor só gera 1 coluna dummy; as outras espécies precisam existir com 0)
nova_cat_normalizada = pd.get_dummies(nova_flor['class'], prefix='class', dummy_na=False, dtype=int)
nova_normalizada = nova_num_normalizada.join(nova_cat_normalizada).reindex(columns=iris_cluster.feature_names_in_, fill_value=0)
print('Nova flor normalizada:')
print(nova_normalizada)

#7. predict: devolve o NÚMERO do cluster da flor
cluster_da_flor = iris_cluster.predict(nova_normalizada)[0]
print('\nCluster da flor:', cluster_da_flor)

#8. Desnormalizar o centróide desse cluster
centroide = pd.DataFrame([iris_cluster.cluster_centers_[cluster_da_flor]], columns=iris_cluster.feature_names_in_)
colunas_num = modelo_normalizador.feature_names_in_

#numéricos: inverse_transform do modelo normalizador
centroide_num = pd.DataFrame(modelo_normalizador.inverse_transform(centroide[colunas_num]), columns=colunas_num)

#categóricos: no centróide as colunas dummy são a PROPORÇÃO de cada espécie no cluster (ex.: 0.9, 0.1, 0)
#o from_dummies só aceita 0/1, então marca 1 na espécie de maior proporção e 0 nas outras
centroide_cat = centroide.drop(columns=colunas_num)
centroide_cat_dummies = centroide_cat.eq(centroide_cat.max(axis=1), axis=0).astype(int)
centroide_cat_natural = pd.from_dummies(centroide_cat_dummies, sep='_')

print('\nCentróide desnormalizado:')
print(centroide_num.join(centroide_cat_natural))
print('\nProporção de cada espécie no cluster:')
print(centroide_cat.round(2))
