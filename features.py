import numpy as np
import pandas as pd

CHEMIN = "data/processed/pesees_propres.csv"  # à adapter selon l'emplacement du fichier
FENETRE = 3   # semaines utilisées pour estimer le niveau de chaque série
HORIZON = 4   # nombre de semaines à prévoir


def s12_by_chrono(chemin=CHEMIN, fenetre=FENETRE, horizon=HORIZON):
    """Prévision de la collecte pour les 4 semaines suivantes (S9 à S12) par site et par catégorie.

    Paramètres :
        chemin : chemin vers le fichier CSV d'entrée (pesees_propres.csv)
        fenetre : nombre de semaines utilisées pour estimer le niveau de chaque série
        horizon : nombre de semaines à prévoir
    Retour :
        un DataFrame avec une ligne par couple site x catégorie et une colonne par semaine
        (S9 à S12), contenant la prévision de collecte en kg.
    """
    # 1. Chargement : une ligne par couple site x catégorie, une colonne par semaine
    df = pd.read_csv(chemin)[["site", "categorie", "semaine", "collecte_kg"]]
    series = df.pivot_table(index=["site", "categorie"], columns="semaine",
                            values="collecte_kg").astype(float)

    # La semaine 1 de Bagnolet est atypique (44 % du niveau des semaines 2 à 4) : on l'écarte
    series_ajustees = series.copy()
    series_ajustees.loc["Bagnolet", 1] = np.nan

    # 2. Prévision finale avec le modèle retenu
    prevision = pd.DataFrame({series.columns.max() + h: moyenne_croissance(series_ajustees, h)
                              for h in range(1, horizon + 1)})

    return prevision

# 2. Modèles : chacun reçoit l'historique connu et l'horizon h, et renvoie une prévision
def decalage(hist, h):
    return hist[hist.columns[-1]]


def moyenne(hist, h, k=FENETRE):
    return hist[hist.columns[-k:]].mean(axis=1)


def tendance_lineaire(hist, h):
    cible = hist.columns.max() + h
    resultat = {}
    for cle, ligne in hist.iterrows():
        ligne = ligne.dropna()
        pente, origine = np.polyfit(ligne.index.astype(float), ligne.values, 1)
        resultat[cle] = origine + pente * cible
    return pd.Series(resultat)


def croissance_hebdo(hist):
    """Croissance log hebdomadaire commune, estimée sur le total à partir de la semaine 2."""
    total = hist.loc[:, hist.columns >= 2].sum()
    return np.polyfit(total.index.astype(float), np.log(total.values), 1)[0]


def moyenne_croissance(hist, h, k=FENETRE):
    # La moyenne est centrée sur le milieu de la fenêtre, d'où le terme (k - 1) / 2
    return moyenne(hist, h, k) * np.exp(croissance_hebdo(hist) * (h + (k - 1) / 2))


# 1. Chargement : une ligne par couple site x catégorie, une colonne par semaine
df = pd.read_csv(CHEMIN)
series = df.pivot_table(index=["site", "categorie"], columns="semaine",
                        values="collecte_kg").astype(float)

# La semaine 1 de Bagnolet est atypique (44 % du niveau des semaines 2 à 4) : on l'écarte
series_ajustees = series.copy()
series_ajustees.loc["Bagnolet", 1] = np.nan




MODELES = {
    "Décalage": decalage,
    "Moyenne 2 sem.": lambda hist, h: moyenne(hist, h, 2),
    "Moyenne 3 sem.": moyenne,
    "Tendance linéaire": tendance_lineaire,
    "Moyenne 3 sem. + croissance": moyenne_croissance,
}

# 3. Validation chronologique : apprentissage jusqu'à T, test sur T + h (jamais de données futures)
derniere = int(series.columns.max())
lignes = []
for T in range(4, derniere):
    hist = series_ajustees.loc[:, series_ajustees.columns <= T]
    for h in range(1, HORIZON + 1):
        if T + h > derniere:
            continue
        for nom, modele in MODELES.items():
            erreur = series[T + h] - modele(hist, h).reindex(series.index)
            lignes.append(pd.DataFrame({"modele": nom, "h": h, "erreur": erreur,
                                        "reel": series[T + h]}))
bt = pd.concat(lignes)
bt["abs"] = bt["erreur"].abs()
bt["pct"] = bt["abs"] / bt["reel"] * 100

print(bt.groupby("modele").agg(MAE_kg=("abs", "mean"), MAPE_pct=("pct", "mean"),
                               biais_kg=("erreur", "mean")).round(1).sort_values("MAE_kg"))
print("\nMAE par horizon :")
print(bt.pivot_table(index="modele", columns="h", values="abs").round(1))

# 4. Prévision finale avec le modèle retenu
print(f"\nCroissance commune : {np.expm1(croissance_hebdo(series_ajustees)):.2%} par semaine")
prevision = pd.DataFrame({derniere + h: moyenne_croissance(series_ajustees, h)
                          for h in range(1, HORIZON + 1)})

print("\nPar site et par catégorie (kg) :")
print(prevision.round(0).astype(int).to_string())
print("\nTotal par site (kg) :")
print(prevision.groupby("site").sum().round(0).astype(int).to_string())
print("\nTotal par catégorie (kg) :")
print(prevision.groupby("categorie").sum().round(0).astype(int).to_string())
print("\nTotal général (kg) :")
print(prevision.sum().round(0).astype(int).to_string())