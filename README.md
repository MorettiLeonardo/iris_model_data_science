# Cluster Iris com KMeans

Agrupa as flores da base `iris.csv` com o KMeans e descobre a qual grupo (cluster) uma flor nova pertence.

Baseado no código das aulas:
- **21/10/2026**: balanceamento, número ótimo de clusters, treino do KMeans e `pickle.dump` (projeto `cluster1`)
- **28/10/2026**: normalização com MinMax e `get_dummies` (`nomalizar.py`)

## Arquivos

| Arquivo | O que é |
|---|---|
| `iris.csv` | Base de dados: 150 flores, 4 medidas (cm) e a espécie (`class`). Separador `;` |
| `iris_cluster.py` | **Treino**: balanceia, normaliza, acha o número ótimo de clusters, treina e salva os modelos |
| `iris_inferencia.py` | **Uso**: carrega os modelos, normaliza uma flor nova, faz o `predict` e desnormaliza o centróide |
| `iris_cluster.model` | Modelo KMeans treinado (gerado pelo `iris_cluster.py`) |
| `modelo_normalizador.model` | Normalizador MinMax treinado (gerado pelo `iris_cluster.py`) |

## Como rodar

Primeiro o treino, depois a inferência (a inferência precisa dos `.model` gerados pelo treino):

```bash
uv run --with pandas --with scikit-learn --with imbalanced-learn --with scipy python iris_cluster.py
uv run --with pandas --with scikit-learn python iris_inferencia.py
```

Ou com pip:

```bash
pip install pandas scikit-learn imbalanced-learn scipy
python iris_cluster.py
python iris_inferencia.py
```

## Passo a passo

### `iris_cluster.py` (treino)

**1. Balancear a base (aula 21/10)**
Separa os atributos (as 4 medidas) da classe (`class`) e aplica o `SMOTE`, que cria exemplos sintéticos das classes com menos registros até todas ficarem do mesmo tamanho. Depois junta tudo de novo com `join`.

> A Iris já vem balanceada (50 flores de cada espécie), então o SMOTE não muda nada aqui. O passo fica no código porque o exercício pede e porque em outras bases ele faz diferença.

**2. Normalizar a base (aula 28/10)**
- **Separar** os numéricos (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`) do categórico (`class`).
- **MinMax** nos numéricos: cada valor vira um número entre 0 e 1 com `(x − mínimo) / (máximo − mínimo)`. Assim nenhuma medida pesa mais que outra no cálculo de distância.
- **get_dummies** no categórico: a coluna `class` vira 3 colunas de 0/1 (`class_Iris-setosa`, `class_Iris-versicolor`, `class_Iris-virginica`).
- **Juntar** os dois com `join` num dataframe único de 7 colunas.
- O normalizador é salvo em `modelo_normalizador.model`, porque depois ele vai ser usado para normalizar a flor nova e desnormalizar o centróide.

**3. Número ótimo de clusters (aula 21/10)**
Treina o KMeans com k de 1 a 99 e calcula a **distorção** de cada um (distância média de cada flor até o centróide mais próximo). Depois aplica o **método do cotovelo**: traça uma reta entre o primeiro e o último ponto da curva, e o k ótimo é o ponto mais distante dessa reta. Na Iris o resultado é **6**.

**4. Treinar e salvar o modelo (aula 21/10)**
Treina o KMeans definitivo com o k ótimo e salva com `pickle.dump` em `iris_cluster.model`.

### `iris_inferencia.py` (uso do modelo)

**5. Nova flor**
Cria uma flor qualquer (pode ser da própria base):
```python
[6.1, 2.8, 4.7, 1.2, 'Iris-versicolor']
```

**6. Normalizar a flor nova (aula 28/10)**
- Numéricos: `modelo_normalizador.transform(...)`. Usa o **mesmo** min/max do treino, por isso o normalizador foi salvo. Nunca chamar `fit` de novo aqui.
- Categórico: `get_dummies` + `reindex`. Uma flor só gera 1 coluna dummy (a da espécie dela); o `reindex` cria as outras com 0 para ficar com as mesmas 7 colunas do treino.

**7. `predict`**
`iris_cluster.predict(...)` mede a distância da flor até cada centróide e devolve o **número** do cluster mais próximo. Para a flor do exemplo: cluster **0**.

> O `predict` devolve o número do cluster, não o centróide. O centróide é obtido com esse número: `iris_cluster.cluster_centers_[cluster]`.

**8. Desnormalizar o centróide**
O centróide está na escala normalizada (0 a 1), então é convertido de volta:
- Numéricos: `modelo_normalizador.inverse_transform(...)` devolve os valores em centímetros.
- Categórico: no centróide as colunas dummy não são 0/1, são a **proporção** de cada espécie no cluster (ex.: 0.9 / 0.1 / 0). Como o `pd.from_dummies` só aceita 0/1, primeiro marca 1 na espécie de maior proporção e 0 nas outras, e aí o `from_dummies` devolve o nome da espécie.

Resultado do exemplo:

```
   sepal_length  sepal_width  petal_length  petal_width            class
0      6.206897     2.972414      4.548276     1.437931  Iris-versicolor
```

Ou seja, a flor nova pertence a um grupo de versicolor com sépala de ~6,2 × 3,0 cm e pétala de ~4,5 × 1,4 cm.

## Dúvidas comuns

**O que são os arquivos `.model`?**
São objetos Python salvos em disco com o `pickle`. O `pickle.dump` grava o objeto treinado e o `pickle.load` o carrega de volta igualzinho. A extensão `.model` é só um nome (poderia ser `.pkl`). Serve para treinar uma vez só e reutilizar sem treinar de novo.
⚠️ Só abra com `pickle.load` arquivos que você mesmo gerou: um pickle malicioso pode executar código na máquina.

**Por que normalizar?**
O KMeans usa distância euclidiana. Sem normalizar, a medida com valores maiores dominaria o cálculo de distância.

**Por que salvar o normalizador e não só o modelo?**
A flor nova precisa ser normalizada com o mesmo mínimo/máximo do treino. Se fizesse `fit` só com ela, os valores sairiam numa escala diferente e o `predict` daria errado. E sem ele não dá para desnormalizar o centróide.

**Por que 6 clusters se só existem 3 espécies?**
O KMeans não sabe quantas espécies existem: o k vem do método do cotovelo, que olha só para as distâncias. Com 6, cada espécie acabou dividida em 2 clusters (por exemplo, versicolor menores e maiores), e nenhum cluster mistura espécies. Um cluster é um grupo de flores parecidas, não uma espécie.

**Diferenças em relação ao `nomalizar.py` da aula**
- A aula chama `fit` e depois `fit_transform`, o que treina o normalizador duas vezes (mesmo resultado). Aqui foi usado só `transform` depois do `fit`.
- O `from_dummies` precisou do passo extra de converter proporções em 0/1 (explicado no passo 8).
