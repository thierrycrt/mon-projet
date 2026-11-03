import pandas as pd
# ***************exercice 3 jour 5  ****************
# # 1. Charger le fichier propre
# df = pd.read_csv("data/processed/pesees_propres.csv")

# # 2. L'état réel des données : lignes, périodes, groupes
# print("Lignes :", len(df))
# print("Périodes :", df["semaine"].nunique())
# print("Groupes :", df["site"].nunique())

# ***************exercice 3 jour 5  ****************
# 1. Charger le fichier propre
df = pd.read_csv("data/processed/pesees_clean.csv")

# 2. L'état réel des données : lignes, périodes, groupes
print("Lignes :", len(df))
print("semaine :", df["semaine"].nunique())
print("site :", df["site"].nunique())
print("Catégories :", df["categorie"].nunique())