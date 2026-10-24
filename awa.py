# Sponsor : responsable d'exploitation · 
# Objectif : augmenter le taux de valorisation · 
# Décision : où cibler la formation au tri

#quoi chercher: mediane de chaque colonne apr site
#

import pandas as pd
import os


print("############### pandas exercice J3 1 ###################")
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

# constat 1: la part du metal dans le taux de collecte est tres faible 8.5%
# constat 2: le taux de valorisation du metal est le plus haut 86.5%
# constat 3 le textile est tres bien valorisé par rapport a la moyenne mondiale valorisé 38.72%

print("############### pandas exercice Jour 3 exo 2 ###################")


import matplotlib.pyplot as plt
import seaborn as sns

# df = pd.read_csv("data/processed/pesees_clean.csv")

ordre = df.groupby("categorie").tx_valorise.median().sort_values(ascending=False).index

sns.set_theme(palette="colorblind")   # couleurs lisibles par tout le monde
plt.figure(figsize=(8, 5))

print("------------------ligne1---------------------------")

sns.barplot(data=df, x="categorie", y="tx_valorise", estimator="median", order=ordre)
plt.title("taux de Valorisation par catégorie en kg")
plt.savefig("figures/comparaison_groupes.png", dpi=150, bbox_inches="tight")
plt.close()   # ferme le graphique avant de dessiner le suivant


print("------------------ligne2---------------------------")

sns.histplot(data=df, x="categorie")
plt.title("Distribution par catégories")
plt.savefig("figures/distribution.png", dpi=150, bbox_inches="tight")
plt.close()   # ferme le graphique avant de dessiner le suivant

print("------------------ligne3---------------------------")

# print("pas de dates")
# print(df["semaine2"].dtype)
# dates = pd.to_datetime(["2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06",
#                         "2026-08-07", "2026-08-08", "2026-08-09"])  
# df_filtre = df[df["semaine2"].isin(dates)]   
    
# # ordre = df.groupby("categorie")["semaine"].max().sort_values(ascending=False).index
# sns.lineplot(data=df_filtre, x="semaine", y="collecte_kg")
# plt.title("Titre qui énonce le message")
# plt.savefig("figures/evolution.png", dpi=150, bbox_inches="tight")
# plt.close()   # ferme le graphique avant de dessiner le suivant

print("------------------ligne4---------------------------")

sns.scatterplot(data=df, x="collecte_kg", y="revendu_kg", hue="categorie")
plt.title("Relation entre deux mesures numériques, recyclé et revendu par catégorie")
plt.savefig("figures/relation.png", dpi=150, bbox_inches="tight")
plt.close()   # ferme le graphique avant de dessiner le suivant


# FIGURE SPONSOR : figures\comparaison_groupes.png — parce qu'on voit clairement due qualité de valorisation tres faible pour le textile et le verre

print("############### pandas exercice Jour 3 exo 3 ###################")


corr = df.corr(numeric_only=True)
print(corr.round(2))

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1)
plt.title("Titre qui énonce le message")
plt.savefig("figures/correlations.png", dpi=150, bbox_inches="tight")
plt.close()

paires = corr.abs().unstack()
paires = paires[paires.index.get_level_values(0) < paires.index.get_level_values(1)]
print("\n=== Les 5 paires les plus corrélées ===")
print(paires.sort_values(ascending=False).head(5))


# === Les 5 paires les plus corrélées ===
# attente_kg      dechetterie_kg    0.936034 -> CORRELATION POSSIBLE:  l'attende se transforme en dechet
# dechetterie_kg  revendu_kg        0.931491 -> aucune correlation
# recycle_kg      valorise_kg       0.900707 -> correlation evidente
# collecte_kg     tx_valorise       0.887331 -> correlation evidente
# attente_kg      collecte_kg       0.873221 -> correlation evidente
#les correlations sont logique et mecanique


print("############### pandas exercice J 3 exo 4 ###################")

# Question : les catégories ont-elles des taux de valorisation différents ?
# Si oui, la formation au tri cible les catégories faibles.
# Si non, le problème se joue ailleurs.


from scipy import stats

# Nombre de lignes par catégorie
tailles = df.groupby("categorie").size()

# Taux de valorisation de chaque ligne
df["tx_valorise"] = (
    (df["revendu_kg"] + df["recycle_kg"]) / df["collecte_kg"] * 100
)


print("\n=== Nombre de lignes par catégorie ===")
print(tailles)

groupes = [g["tx_valorise"].dropna() for _, g in df.groupby("categorie")]

if tailles.min() < 2:
    print("Test impossible : au moins une catégorie n'a qu'une seule ligne.")
else:
    stat, p = stats.f_oneway(*groupes)
    print("\n=== Test f_oneway : taux de valorisation selon la catégorie ===")
    print(
        f"Test choisi : f_oneway, parce que {len(groupes)} catégories "
        f"de {tailles.min()} à {tailles.max()} lignes chacune."
    )
    print(f"valeur-p = {p:.3f}")

    # Conclusion en français métier
    if p < 0.05:
        print("On peut affirmer que les taux de valorisation diffèrent selon la catégorie.")
        print("Conséquence : cibler la formation au tri sur les catégories les plus faibles.")
    else:
        print("On ne peut pas affirmer que les taux de valorisation diffèrent selon la catégorie.")
        print("Conséquence : l'écart vu sur le graphique peut n'être que du bruit ; chercher ailleurs.")

    

# on ne peut rien afficher les chiffres ne sont pas suffisant pour les analyser

print("############### pandas exercice Jour 3 exo 5 ###################")
print(df)

df["z"] = df.groupby("categorie")["tx_valorise"].transform(
    lambda s: (s - s.mean()) / s.std()
)
print(df["z"])
print(df[df["z"].abs() > 2])                       # les anomalies
print(df.loc[df["z"].dropna().abs().nlargest(5).index])     # les 5 plus marquées, même si la liste du dessus est vide

# 0  Montreuil        8   Textile         1206       249.0       221.0             635         101                False               False               False                   False               False        470.0    38.971808  0.707107
# 6  Montreuil        8   Textile         1300       250.0       250.0             600         200                False               False               False                   False               False        500.0    38.461538 -0.707107
#  z le plus haut 0.7 de difference entre deux lignes copier collé avec modif de chiffres

print("############### pandas exercice Jour 3 exo 6 ###################")
