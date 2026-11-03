import pandas as pd
import numpy as np

print("#-------------------------- Jours 6 exo 2-------------------------")


df = pd.read_csv("data/processed/pesees_propres.csv")

# 1. Filtrer Pantin + Bois, trier par semaine
serie = (
    df[(df["site"].str.lower() == "pantin") & (df["categorie"].str.lower() == "bois")]
    .sort_values("semaine")
    .set_index("semaine")["collecte_kg"]
)


# print("\n=== Scores over the last 3 weeks for the Pantin site in wood collection ===")
# 2. Les deux baselines (shift(1) = on n'utilise que le passé, pas de fuite de données)
baselines = pd.DataFrame({
    "reel":          serie,
    "naive":         serie.shift(1),
    "moy_mobile_3":  serie.rolling(3).mean().shift(1),
})

# 3. Garder les 3 dernières semaines
dernieres = baselines.tail(3)
print(dernieres)

# 4. Erreur moyenne absolue (MAE) de chaque baseline sur ces 3 semaines
mae = (dernieres[["naive", "moy_mobile_3"]]
       .sub(dernieres["reel"], axis=0).abs().mean())
print(mae)   # naive 21.33 kg | moy_mobile_3 16.00 kg

# 5. Prévision pour la semaine suivante (semaine 9)
print("Naïve S9 :", serie.iloc[-1])                 # 431 kg (valeur de S8)
print("Moyenne mobile S9 :", serie.tail(3).mean())  # 426.67 kg (moyenne S6-S8)

print("#-------------------------- Jours 6 exo 3 -------------------------")
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/processed/pesees_propres.csv").sort_values(["site", "categorie", "semaine"])

# ── 2. Variables connues au moment de prédire (toujours décalées) ──
g = df.groupby(["site", "categorie"])["collecte_kg"]
for k in (1, 2, 3):
    df[f"lag_{k}"] = g.shift(k)
df["moy_3"]   = g.transform(lambda s: s.shift(1).rolling(3).mean())
df["moy_exp"] = g.transform(lambda s: s.shift(1).expanding().mean())
df = df.dropna(subset=["lag_1", "lag_2", "lag_3", "moy_3"]).reset_index(drop=True)

num = ["lag_1", "lag_2", "lag_3", "moy_3", "moy_exp", "semaine"]
cat = ["site", "categorie"]
X, y = df[num + cat], df["collecte_kg"]

# Encodage dans le pipeline : le scaler est ajusté sur le train uniquement
prep = ColumnTransformer([
    ("num", StandardScaler(), num),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
])

# ── 3. Découpage temporel : entraînement S4 à S7, test S8 ──
train = df["semaine"].between(4, 7)   # 72 lignes
test  = df["semaine"] == 8            # 18 lignes (6 catégories × 3 sites)
pb    = test & (df["site"] == "Pantin") & (df["categorie"] == "Bois")

# ── Modèles ──
modeles = {
    "Ridge": Ridge(alpha=1.0),
    "MLP": TransformedTargetRegressor(            # y normalisé, sinon le MLP diverge
        regressor=MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1.0,
                               max_iter=5000, random_state=42),
        transformer=StandardScaler(),
    ),
}

print(f"{'Modèle':<15}{'MAE toutes séries':>20}{'MAE Pantin/Bois':>18}")
print(f"{'Naïve':<15}{mean_absolute_error(y[test], df.loc[test, 'lag_1']):>20.1f}"
      f"{mean_absolute_error(y[pb], df.loc[pb, 'lag_1']):>18.1f}")
print(f"{'Moy. mobile 3':<15}{mean_absolute_error(y[test], df.loc[test, 'moy_3']):>20.1f}"
      f"{mean_absolute_error(y[pb], df.loc[pb, 'moy_3']):>18.1f}")

for nom, modele in modeles.items():
    pipe = Pipeline([("prep", prep), ("model", modele)]).fit(X[train], y[train])
    print(f"{nom:<15}{mean_absolute_error(y[test], pipe.predict(X[test])):>20.1f}"
          f"{mean_absolute_error(y[pb], pipe.predict(X[pb])):>18.1f}")



print("#-------------------------- Jours 6 exo 4-------------------------")

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/processed/pesees_propres.csv").sort_values(["site", "categorie", "semaine"])

# Variables connues au moment de prédire
g = df.groupby(["site", "categorie"])["collecte_kg"]
for k in (1, 2, 3):
    df[f"lag_{k}"] = g.shift(k)
df["moy_3"]   = g.transform(lambda s: s.shift(1).rolling(3).mean())
df["moy_exp"] = g.transform(lambda s: s.shift(1).expanding().mean())
df = df.dropna(subset=["lag_1", "lag_2", "lag_3", "moy_3"]).reset_index(drop=True)

num = ["lag_1", "lag_2", "lag_3", "moy_3", "moy_exp", "semaine"]
cat = ["site", "categorie"]
X, y = df[num + cat], df["collecte_kg"]

train = df["semaine"].between(4, 7)
test  = df["semaine"] == 8
pb    = test & (df["site"] == "Pantin") & (df["categorie"] == "Bois")

# Fit / predict
modele = Pipeline([
    ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)],
                               remainder="passthrough")),
    ("model", LinearRegression()),
])
modele.fit(X[train], y[train])
pred_train = modele.predict(X[train])
pred_test  = modele.predict(X[test])

# MAE
print("LinearRegression  train :", round(mean_absolute_error(y[train], pred_train), 1))  # 33.9
print("LinearRegression  test  :", round(mean_absolute_error(y[test],  pred_test), 1))   # 38.9
print("LinearRegression  Pantin/Bois :",
      round(mean_absolute_error(y[pb], modele.predict(X[pb])), 1))                      # 40.9 (prévu 471.9, réel 431)

# Baselines sur le même test
print("Naïve             test  :", round(mean_absolute_error(y[test], df.loc[test, "lag_1"]), 1))  # 56.7
print("Moy. mobile 3     test  :", round(mean_absolute_error(y[test], df.loc[test, "moy_3"]), 1))  # 37.5

# ─────────────────────────────────────────────────────────────────────────────
# VERDICT (test = semaine 8, 18 lignes)

# Toutes séries : LinearRegression 38.9 kg vs moyenne mobile 37.5 kg
#   → MOINS BIEN que la meilleure baseline (+1.4 kg). L'écart est faible
#     sur 18 lignes, mais le modèle ne fait pas mieux qu'une simple moyenne.
#   → Mieux que la baseline naïve (56.7 kg), qui n'est pas la référence à battre.

# Pantin / Bois : LinearRegression 40.9 kg vs moyenne mobile 5.0 kg
#   → NETTEMENT MOINS BIEN (une seule prévision : 471.9 kg pour 431 réels).

# Train 33.9 kg vs test 38.9 kg : léger sur-apprentissage
#   (15 variables pour 72 lignes, sans régularisation).
#   Ridge (alpha=1) faisait 33.8 kg en test : la régularisation aide ici.

# Conclusion : LinearRegression n'apporte rien par rapport à la moyenne
# mobile sur 3 semaines. À ne pas retenir en l'état.
# ─────────────────────────────────────────────────────────────────────────────



print("#-------------------------- Jours 6 exo 5-------------------------")

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("data/processed/pesees_propres.csv").sort_values(["site", "categorie", "semaine"])

g = df.groupby(["site", "categorie"])["collecte_kg"]
for k in (1, 2, 3):
    df[f"lag_{k}"] = g.shift(k)
df["moy_3"]   = g.transform(lambda s: s.shift(1).rolling(3).mean())
df["moy_exp"] = g.transform(lambda s: s.shift(1).expanding().mean())
df = df.dropna(subset=["lag_1", "lag_2", "lag_3", "moy_3"]).reset_index(drop=True)

num = ["lag_1", "lag_2", "lag_3", "moy_3", "moy_exp", "semaine"]
cat = ["site", "categorie"]
X, y = df[num + cat], df["collecte_kg"]

train = df["semaine"].between(4, 7)
test  = df["semaine"] == 8

modele = Pipeline([
    ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat)],
                               remainder="passthrough")),
    #("model", DecisionTreeRegressor(random_state=0)),   # aucune limite de profondeur
    ("model", DecisionTreeRegressor(max_depth=3, random_state=0)), # profondeur max 3 (random_state = choix des coupures)
])
modele.fit(X[train], y[train])

# Valeurs : max_depth=3  |  (sans limite)
print("Arbre  train :", round(mean_absolute_error(y[train], modele.predict(X[train])), 1))  # 39.0  | (0.0)
print("Arbre  test  :", round(mean_absolute_error(y[test],  modele.predict(X[test])), 1))   # 60.2  | (52.5)
arbre = modele.named_steps["model"]
print("Profondeur :", arbre.get_depth(), "| Feuilles :", arbre.get_n_leaves())             # 3 | 8 | (9 | 72)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTAT

# Train 0.0 kg / test 52.5 kg : sur-apprentissage total.
#   72 feuilles pour 72 lignes d'entraînement → une feuille par ligne.
#   L'arbre a appris le train par cœur et ne généralise pas.

# Comparaison au test (semaine 8) :
#   Moyenne mobile 3   37.5 kg   ← meilleure baseline
#   LinearRegression   38.9 kg
#   Arbre sans limite  52.5 kg   → NETTEMENT MOINS BIEN, presque aussi mauvais
#                                  que la baseline naïve (56.7 kg)

# Instable : selon random_state, la MAE test varie de 44.1 à 55.6 kg.

# Conclusion : un MAE d'entraînement à 0 n'est pas une bonne nouvelle, c'est le
# signal d'alarme. À ne pas retenir sans contraindre l'arbre (max_depth,
# min_samples_leaf).

# CONSTAT max_depth=3
# Train 39.0 kg / test 60.2 kg : écart réduit (21.2 kg contre 52.5 kg sans limite),
#   le sur-apprentissage est corrigé, mais le test n'est pas meilleur.
#   Avec random_state 1, 2 ou 3 : test 46.8 kg.
# Contre la baseline (moyenne mobile 37.5 kg) : MOINS BIEN dans tous les cas.
# 8 feuilles = 8 valeurs possibles pour 18 séries de niveaux très différents :
#   l'arbre est passé du sur-apprentissage au sous-apprentissage.
# ─────────────────────────────────────────────────────────────────────────────