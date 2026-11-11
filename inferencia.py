# MÓDULO DE INFERÊNCIA - a qual grupo pertence uma nova residência
import pickle

import pandas as pd

# 1) Carregando o modelo e o normalizador salvos no treinamento
with open("modelo_kmeans.pkl", "rb") as arquivo:
    modelo = pickle.load(arquivo)
with open("normalizador.pkl", "rb") as arquivo:
    normalizador = pickle.load(arquivo)

colunas_quantitativas = ["CRIM", "ZN", "INDUS", "NOX", "RM", "AGE", "DIS", "RAD",
                         "TAX", "PTRATIO", "B", "LSTAT", "MEDV"]
coluna_categorica = "CHAS"

descricoes = {
    "CRIM": "taxa de criminalidade per capita",
    "ZN": "% de terrenos residenciais grandes",
    "INDUS": "% de area comercial nao varejista",
    "CHAS": "a margem do rio Charles? (1 = sim, 0 = nao)",
    "NOX": "concentracao de oxidos nitricos",
    "RM": "numero medio de comodos",
    "AGE": "% de imoveis construidos antes de 1940",
    "DIS": "distancia ate centros de emprego",
    "RAD": "indice de acesso a rodovias",
    "TAX": "imposto sobre a propriedade",
    "PTRATIO": "razao aluno/professor",
    "B": "indice B da populacao",
    "LSTAT": "% da populacao de baixa renda",
    "MEDV": "valor mediano do imovel (US$ 1000)",
}

# 2) Recebendo os dados da nova residência
print("Digite os dados da nova residencia:")
nova = {}
for coluna in descricoes:
    nova[coluna] = float(input(f"{coluna} - {descricoes[coluna]}: ").replace(",", "."))
nova = pd.DataFrame([nova])

# 3) Normalizando a nova residência do mesmo jeito que no treinamento:
# quantitativas com o normalizador salvo, CHAS mantida como 0/1
nova_norm = pd.DataFrame(normalizador.transform(nova[colunas_quantitativas]),
                         columns=colunas_quantitativas)
nova_norm[coluna_categorica] = int(nova[coluna_categorica].iloc[0])

# 4) Inferindo o grupo
grupo = modelo.predict(nova_norm)[0]
print(f"\nA nova residencia pertence ao GRUPO {grupo}")

# 5) Desnormalizando o centroide do grupo
centroide = pd.DataFrame([modelo.cluster_centers_[grupo]], columns=nova_norm.columns)
centroide_original = pd.DataFrame(normalizador.inverse_transform(centroide[colunas_quantitativas]),
                                  columns=colunas_quantitativas)
# No centroide, CHAS é a proporção de residências do grupo à margem do rio.
# Para voltar à categoria original, usa-se a categoria predominante.
proporcao_rio = centroide[coluna_categorica].iloc[0]
centroide_original[coluna_categorica] = 1 if proporcao_rio >= 0.5 else 0

# 6) Exibindo a nova residência ao lado do centroide desnormalizado
colunas = list(descricoes)
resultado = pd.DataFrame({
    "Nova residencia": nova[colunas].iloc[0],
    f"Centroide do grupo {grupo}": centroide_original[colunas].iloc[0].round(2),
})
print("\nDados desnormalizados do centroide do grupo:")
print(resultado)
