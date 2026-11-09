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

CHEMIN_ML = CHEMIN


def variables(df):
    """Variable de chrono_collecte : moyenne 3 sem. x croissance commune (h = 1).
    Pour la ligne de la semaine S, tout est calculé avec les semaines < S :
      - moy_3      : moyenne de S-3..S-1 (Bagnolet S1 écartée, comme dans s12_by_chrono)
      - croissance : pente log du total des semaines 2..S-1
      - prev_chrono = moy_3 * exp(croissance * (h + (k - 1) / 2)), h = 1, k = 3
    """
    df = df.sort_values(["site", "categorie", "semaine"]).copy()
    ajuste = df["collecte_kg"].where(~((df["site"] == "Bagnolet") & (df["semaine"] == 1)))
    df["moy_3"] = ajuste.groupby([df["site"], df["categorie"]]).transform(
        lambda s: s.shift(1).rolling(window=FENETRE, min_periods=2).mean())
    total = df.groupby("semaine")["collecte_kg"].sum()
    croissance = {}
    for s in total.index:
        passe = total.loc[2:s - 1]
        if len(passe) >= 2:
            croissance[s] = np.polyfit(passe.index.astype(float), np.log(passe.values), 1)[0]
    df["croissance"] = df["semaine"].map(croissance)
    df["prev_chrono"] = df["moy_3"] * np.exp(df["croissance"] * (1 + (FENETRE - 1) / 2))
    return df


VARIABLES = ["prev_chrono"]


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
