import pandas as pd

CHEMIN = "data/processed/pesees_propres.csv"  # à adapter selon l'emplacement du fichier
FENETRE = 3  # nombre de semaines de la moyenne glissante


def s9_by_rolling_mean(chemin, fenetre=3):
    
    #"""Prévision de la semaine 9 par moyenne glissante sur les 3 semaines précédentes.

    #Paramètres :
    #    chemin : chemin vers le fichier CSV d'entrée (par défaut "pesees_propres.csv")
    #    fenetre : nombre de semaines pour la moyenne glissante (par défaut 3)

    CIBLES = ["collecte_kg", "revendu_kg", "recycle_kg", "dechetterie_kg", "attente_kg"]
    CLES = ["site", "categorie"]

    # 1. Chargement et tri chronologique
    df = pd.read_csv(CHEMIN).sort_values(CLES + ["semaine"])

    # 2. Agrégat glissant sur l'historique : moyenne des FENETRE semaines PRÉCÉDENTES
    #    (le shift(1) exclut la semaine à prévoir et évite toute fuite de données)
    previsions = df.groupby(CLES)[CIBLES].transform(
        lambda s: s.shift(1).rolling(window=FENETRE, min_periods=FENETRE).mean()
    )

    # 3. Prochaine collecte = moyenne des FENETRE dernières semaines observées
    derniere_semaine = df["semaine"].max()
    prochaine = (
        df[df["semaine"] > derniere_semaine - FENETRE]
        .groupby(CLES, as_index=False)[CIBLES].mean()
        .assign(semaine=derniere_semaine + 1)
    )

    # 4. Affichage par site et par catégorie (+ totaux)
    print(prochaine.set_index(CLES)[CIBLES].round(0).astype(int).to_string())
    print("\nTotal par site :")
    print(prochaine.groupby("site")[CIBLES].sum().round(0).astype(int).to_string())
    print("\nTotal par catégorie :")
    print(prochaine.groupby("categorie")[CIBLES].sum().round(0).astype(int).to_string())
    print("\nTotal général :")
    print(prochaine[CIBLES].sum().round(0).astype(int).to_string())

    # 5. Fiabilité : erreur par résultat, à partir de la semaine 5
    #    (la semaine 4 est exclue : sa fenêtre contient la semaine 1, atypique à Bagnolet)
    masque = df["semaine"] >= 5
    erreurs = (df[CIBLES] - previsions)[masque]
    fiabilite = pd.DataFrame({
        "MAE_kg": erreurs.abs().mean(),
        "MAPE_%": (erreurs.abs() / df.loc[masque, CIBLES]).mean() * 100,
        "biais_kg": erreurs.mean(),
    }).round(1)
    print("\nFiabilité (semaines 5 à 8) :")
    print(fiabilite.to_string())
    return "ok"

s9_by_rolling =s9_by_rolling_mean(CHEMIN, FENETRE)
