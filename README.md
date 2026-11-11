# Clusterização Boston Housing com KMeans

Agrupa as residências da base Boston Housing com KMeans e indica a qual grupo uma nova residência pertence.

## Como rodar

Execute os comandos dentro desta pasta:

```bash
pip install -r requirements.txt
python treinamento.py
python inferencia.py
```

`treinamento.py` treina e salva o modelo. `inferencia.py` classifica uma nova residência.

## Módulo de treinamento (`treinamento.py`)

1. **Normalização com cuidado sobre categóricos e quantitativos**
   - **Quantitativas** (CRIM, ZN, INDUS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT, MEDV): `MinMaxScaler`, que deixa cada coluna entre 0 e 1.
   - **Categórica** (CHAS, margem do rio Charles): não é normalizada. Ela tem só duas categorias e já está codificada como 0/1, que equivale a uma coluna dummy na mesma escala das quantitativas normalizadas.
   - Antes de normalizar, os valores faltantes são preenchidos: mediana nas quantitativas e moda em CHAS.
2. **Balanceamento:** não é feito, conforme o enunciado.
3. **Método do cotovelo com distâncias:**
   - Para k de 1 a 10, calcula a distorção, que é a distância média de cada ponto ao centroide mais próximo.
   - O k ótimo é o ponto da curva mais distante da reta que liga o primeiro ao último ponto. **Resultado: k = 3.**
   - O gráfico fica em `cotovelo.png`.
4. **Treinar e salvar:** o modelo é salvo em `modelo_kmeans.pkl` e o normalizador em `normalizador.pkl`.

## Módulo de inferência (`inferencia.py`)

1. Recebe pelo teclado os dados de uma nova residência.
2. Normaliza a residência com o normalizador salvo. CHAS continua 0/1, como no treino.
3. Infere o grupo com `modelo.predict`.
4. Exibe o centroide do grupo desnormalizado:
   - Quantitativas: `normalizador.inverse_transform`.
   - CHAS: categoria predominante no grupo.

## Grupos encontrados (k = 3)

| Grupo | Residências | Perfil (centroide) |
|------:|------------:|--------------------|
| 0 | 178 | Pouca criminalidade (CRIM 0,1), poucos moradores de baixa renda (LSTAT 7,6%), valor mais alto (MEDV 27,4) |
| 1 | 136 | Região central: acesso a rodovias máximo (RAD 24), imposto alto (TAX 667), mais criminalidade (CRIM 11,9), valor mais baixo (MEDV 16,2) |
| 2 | 192 | Intermediário (CRIM 0,6, LSTAT 13,3%, MEDV 22,5) |
