import pandas as pd


print("############### pandas exercice 3.1 ###################")

#cmd git switch -C awa

# Chargement du fichier CSV
df = pd.read_csv("data/brut/pesees.csv")

# Vue d'ensemble de toutes les colonnes numériques
print(df.describe())

# Médiane de la mesure principale pour chaque groupe (une ligne par groupe)
print(df.groupby("colonne_de_groupe")["mesure_principale"].median())


