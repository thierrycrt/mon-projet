"""Variables du modèle de prévision des kg partis en déchetterie (J7)."""
import pandas as pd

def charger(chemin="data/processed/pesees_propres.csv"):
    """Charge les pesées et les trie : les décalages dépendent de l'ordre des semaines."""
    df = pd.read_csv(chemin)
    return df.sort_values(["site", "categorie", "semaine"]).reset_index(drop=True)

def ajouter_decalages(df):
    """Décalages : la valeur de la semaine d'avant. Connue en début de semaine S."""
    g = df.groupby(["site", "categorie"])
    df["lag1"] = g["dechetterie_kg"].shift(1)              # déchetterie de S-1
    df["collecte_lag1"] = g["collecte_kg"].shift(1)        # collecte de S-1
    return df

def ajouter_moyennes(df):
    """Agrégats glissants : moyenne des 3 semaines d'avant, sans la semaine S (shift avant rolling)."""
    g = df.groupby(["site", "categorie"])["dechetterie_kg"]
    df["mean3"] = g.transform(lambda s: s.shift(1).rolling(3).mean())
    return df

DIMENSIONS = ["site", "categorie"]   # connues à l'avance : encodées dans le pipeline (exercice 2)

CANDIDATS = {
    "feat-lag1": ["lag1"],                                  # minimal
    "feat-mean3": ["lag1", "mean3"],                        # le pari
    "feat-collecte": ["lag1", "mean3", "collecte_lag1"],    # enrichi
}

def construire(chemin="data/processed/pesees_propres.csv"):
    """Toutes les variables, puis on retire les semaines sans historique (1 à 3)."""
    df = ajouter_moyennes(ajouter_decalages(charger(chemin)))
    return df.dropna(subset=["lag1", "mean3", "collecte_lag1"]).reset_index(drop=True)