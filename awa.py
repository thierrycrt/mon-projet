# Sponsor : responsable d'exploitation · 
# #Objectif : augmenter le taux de valorisation · 
# #Décision : où cibler la formation au tri

#quoi chercher: mediane de chaque colonne apr site
#

import pandas as pd

df = pd.read_csv("data/processed/pesees_clean.csv")

print("=== Aperçu des données ===")
print(df.head)


print("\n=== Dimensions du jeu de données ===")
print(df.shape)

print("\n=== Statistiques descriptives ===")
print(df.describe())

# 3. Médiane de la collecte par site
print("\n=== Médiane de collecte par site ===")
print(df.groupby("site").collecte_kg.median())

print("\n=== Médiane de revente par site ===")
print(df.groupby("site").revendu_kg.median())

print("\n=== Médiane de recyclage par site ===")
print(df.groupby("site").recycle_kg.median())

print("\n=== Médiane de dechet par site ===")
print(df.groupby("site").dechetterie_kg.median())

print("\n=== Médiane d' attente par site ===")
print(df.groupby("site").attente_kg.median())

collect_total = df.collecte_kg.sum()

part_categorie = (
    df.groupby("categorie").collecte_kg.sum() 
    / collect_total *100
    ).sort_values(ascending=False)



print("\n=== Part de chaque catégorie dans la collecte (%) ===")
print(part_categorie.round(2))

# 6. Calculer le taux de valorisation
# Hypothèse : revendu + recycle = volumes valorisés
df["valorise_kg"] = df.revendu_kg + df.recycle_kg
df["tx_valorise"] = df.valorise_kg / df.collecte_kg * 100

print("\n=== Taux de valorisation (%) ===")

print(
    df.groupby("categorie").tx_valorise
    .mean()
    .round(2)
    )

print(df.valorise_kg.sum())
print(collect_total)

global_sum=df.valorise_kg.sum() / collect_total * 100
print(f"\nTaux global de valorisation : {global_sum:.1f} %")
