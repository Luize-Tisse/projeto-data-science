# MÓDULO DE TREINAMENTO - KMeans na base Boston Housing
import math
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# 1) Leitura da base
dados = pd.read_csv("HousingData (1).csv")
print("Formato da base:", dados.shape)

# 2) Separação entre dados categóricos e quantitativos
# CHAS (1 = imóvel à margem do rio Charles, 0 = não) é a única categórica.
# Todas as outras colunas são quantitativas.
coluna_categorica = "CHAS"
colunas_quantitativas = [c for c in dados.columns if c != coluna_categorica]

# 3) Tratamento dos valores faltantes (o KMeans não aceita valores vazios)
print("\nValores faltantes:")
print(dados.isna().sum())
# quantitativas -> mediana | categórica -> moda (valor mais frequente)
for coluna in colunas_quantitativas:
    dados[coluna] = dados[coluna].fillna(dados[coluna].median())
dados[coluna_categorica] = dados[coluna_categorica].fillna(dados[coluna_categorica].mode()[0])

# 4) Normalização
# Quantitativas: MinMaxScaler, que deixa cada coluna entre 0 e 1.
# Categórica: não é normalizada. CHAS só tem duas categorias e já está
# codificada como 0 e 1 (equivale a uma coluna dummy), na mesma escala das
# quantitativas normalizadas.
normalizador = MinMaxScaler()
dados_norm = pd.DataFrame(normalizador.fit_transform(dados[colunas_quantitativas]),
                          columns=colunas_quantitativas)
dados_norm[coluna_categorica] = dados[coluna_categorica].astype(int)
print("\nBase normalizada:")
print(dados_norm.head())

# 5) Número ótimo de grupos pelo método do cotovelo
# Distorção = média das distâncias de cada ponto ao seu centroide mais próximo
K = range(1, 11)
distorcoes = []
for k in K:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(dados_norm)
    distorcoes.append(sum(np.min(cdist(dados_norm, kmeans.cluster_centers_, "euclidean"),
                                 axis=1)) / dados_norm.shape[0])

# O cotovelo é o ponto da curva com a maior distância até a reta que liga
# o primeiro ponto (k = 1) ao último ponto (k = 10)
x0, y0 = K[0], distorcoes[0]
xn, yn = K[-1], distorcoes[-1]
distancias = []
for i in range(len(distorcoes)):
    x, y = K[i], distorcoes[i]
    numerador = abs((yn - y0) * x - (xn - x0) * y + xn * y0 - yn * x0)
    denominador = math.sqrt((yn - y0) ** 2 + (xn - x0) ** 2)
    distancias.append(numerador / denominador)

numero_otimo = K[distancias.index(max(distancias))]
print("\nDistancia de cada k ate a reta:")
for k, d in zip(K, distancias):
    print(f"k = {k:2d}  ->  {d:.4f}")
print("Numero otimo de grupos:", numero_otimo)

plt.plot(K, distorcoes, "o-")
plt.plot([x0, xn], [y0, yn], "--", color="gray")
plt.plot(numero_otimo, distorcoes[numero_otimo - 1], "o", color="red", markersize=10)
plt.title(f"Método do cotovelo (k ótimo = {numero_otimo})")
plt.xlabel("Número de grupos (k)")
plt.ylabel("Distorção")
plt.xticks(K)
plt.savefig("cotovelo.png")
plt.close()

# 6) Treinamento do modelo final com o número ótimo de grupos
modelo = KMeans(n_clusters=numero_otimo, n_init=10, random_state=42).fit(dados_norm)
print("\nQuantidade de residencias em cada grupo:")
print(pd.Series(modelo.labels_).value_counts().sort_index())

# 7) Salvando o modelo e o normalizador (a inferência precisa dos dois)
with open("modelo_kmeans.pkl", "wb") as arquivo:
    pickle.dump(modelo, arquivo)
with open("normalizador.pkl", "wb") as arquivo:
    pickle.dump(normalizador, arquivo)
print("\nModelo salvo em modelo_kmeans.pkl e normalizador em normalizador.pkl")
