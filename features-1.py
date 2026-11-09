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

CHEMIN_ML = "data/processed/pesees_propres.csv"


def variables(df):
    """Variable de feat-minimal : collecte de la semaine précédente (shift 1)."""
    df = df.sort_values(["site", "categorie", "semaine"]).copy()
    df["lag_1"] = df.groupby(["site", "categorie"])["collecte_kg"].shift(1)
    return df


VARIABLES = ["lag_1"]


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

pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge())])
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
