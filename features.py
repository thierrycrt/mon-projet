#-------------------------- Jours 7 : shift -------------------------"
# Prévision de la semaine 9 pour les 18 séries, avec :
#   - la baseline moyenne mobile 3 (meilleure sur Pantin/Bois)
#   - la variante 2 (meilleure sur l'ensemble des séries, MAE 38.1 vs 41.1)
# Rien n'est connu de S9 : on ajoute 18 lignes vides et on ne calcule leurs variables qu'à partir de S6, S7, S8 (shift >= 1, testé sans fuite). 

import pandas as pd



def s9_by_shift(CHEMIN):

    # 1. Chargement et tri chronologique
    df = pd.read_csv(CHEMIN).sort_values(["site", "categorie", "semaine"])

    # 2. Décalage : la prévision d'une semaine = la collecte de la semaine précédente
    df["prevision_kg"] = df.groupby(["site", "categorie"])["collecte_kg"].shift(1)

    # 3. Prévision de la prochaine collecte (semaine 9) = dernière valeur observée (semaine 8)
    derniere_semaine = df["semaine"].max()
    prochaine = (
        df[df["semaine"] == derniere_semaine]
        .assign(semaine=derniere_semaine + 1)
        .rename(columns={"collecte_kg": "prevision_kg_s9"})
        [["site", "categorie", "semaine", "prevision_kg_s9"]]
    )

    # 4. Affichage en tableau site x catégorie
    tableau = prochaine.pivot(index="categorie", columns="site", values="prevision_kg_s9")
    tableau.loc["Total"] = tableau.sum()
    tableau["Total"] = tableau.sum(axis=1)
    print(tableau)

    # 5. Fiabilité : erreur du décalage sur l'historique (semaines 2 à 8)
    test = df.dropna(subset=["prevision_kg"]).copy()
    test["erreur_abs"] = (test["collecte_kg"] - test["prevision_kg"]).abs()
    print(f"\nMAE  : {test['erreur_abs'].mean():.1f} kg")
    print(f"MAPE : {(test['erreur_abs'] / test['collecte_kg']).mean():.1%}")
    return "ok"

by_shift= s9_by_shift("data/processed/pesees_propres.csv" ) 