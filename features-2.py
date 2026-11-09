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


CHEMIN_ML = "data/processed/pesees_propres.csv"


def variables(df):
    """Variable de feat-mean : moyenne glissante des 3 semaines PRÉCÉDENTES."""
    df = df.sort_values(["site", "categorie", "semaine"]).copy()
    df["moy_3"] = df.groupby(["site", "categorie"])["collecte_kg"].transform(
        lambda s: s.shift(1).rolling(window=3, min_periods=3).mean())
    return df


VARIABLES = ["moy_3"]


# ═════════════════════════════════════════════════════════════════════════════
# Jours 7 — La variante passée dans Pipeline(StandardScaler, Ridge)
#   Découpage temporel identique à modele.py : train S4 à S7, test S8 (18 lignes),
#   le même sur les trois branches pour que les MAE soient comparables.
#   Le pipeline entier est ajusté par UN SEUL fit(X_train, y_train) :
#   le scaler calcule moyenne / écart-type sur le train uniquement,
#   predict(X_test) applique ensuite scaler + Ridge sans rien réapprendre.
# ═════════════════════════════════════════════════════════════════════════════
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

print("#-------------------------- Jours 7 : Pipeline Ridge -------------------------")

brut = pd.read_csv(CHEMIN_ML)
df_ml = variables(brut).dropna(subset=VARIABLES)

train = df_ml["semaine"].between(4, 7)
test = df_ml["semaine"] == 8
X, y = df_ml[VARIABLES], df_ml["collecte_kg"]

pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1))])  # même alpha qu'à l'évaluation
pipe.fit(X[train], y[train])                      # train seul
pred_test = pipe.predict(X[test])                 # scaler + Ridge appliqués au test

mae_train = mean_absolute_error(y[train], pipe.predict(X[train]))
mae_test = mean_absolute_error(y[test], pred_test)
mae_regle = mean_absolute_error(y[test], df_ml.loc[test, VARIABLES[0]])
print(f"Variables          : {VARIABLES}")
print(f"Lignes train/test  : {train.sum()} / {test.sum()}")
print(f"MAE train          : {mae_train:.1f} kg")
print(f"MAE test (S8)      : {mae_test:.1f} kg")
print(f"Règle brute {VARIABLES[0]} sur S8 : {mae_regle:.1f} kg")

# ── Contrôles anti-fuite ─────────────────────────────────────────────────────
# 1. Découpage chronologique : tout le train est antérieur au test.
assert df_ml.loc[train, "semaine"].max() < df_ml.loc[test, "semaine"].min()
assert not (train & test).any()

# 2. Le scaler du pipeline n'a vu que le train : mêmes statistiques que X_train,
#    différentes de celles du jeu complet (train + test).
sc = pipe.named_steps["scaler"]
assert sc.n_samples_seen_ == train.sum()
assert np.allclose(sc.mean_, X[train].mean()) and np.allclose(sc.scale_, X[train].std(ddof=0))
complet = X[train | test]
assert not np.allclose(sc.mean_, complet.mean())

# 3. Les variables ne lisent jamais la semaine prédite : on brouille la collecte
#    de S8, les variables de S4 à S8 doivent rester identiques au kg près.
brouille = brut.assign(collecte_kg=brut["collecte_kg"].where(brut["semaine"] != 8, -1e6))
cle = ["site", "categorie", "semaine"]
avant = variables(brut).set_index(cle).loc[lambda d: d.index.get_level_values("semaine") >= 4, VARIABLES]
apres = variables(brouille).set_index(cle).loc[avant.index, VARIABLES]
pd.testing.assert_frame_equal(avant, apres)

# 4. Le test n'influence pas le modèle : y_test brouillé, refit -> mêmes coefficients.
y_brouille = y.where(~test, -1e6)
pipe_b = Pipeline([("scaler", StandardScaler()), ("model", Ridge())]).fit(X[train], y_brouille[train])
assert np.allclose(pipe_b.named_steps["model"].coef_, pipe.named_steps["model"].coef_)

print("Anti-fuite : OK (découpage chronologique, scaler = stats du train seul,"
      " variables insensibles à S8, modèle insensible au test)")
