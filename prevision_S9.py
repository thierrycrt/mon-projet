print("#-------------------------- Jours 6 : prévision S9 -------------------------")
# Prévision de la semaine 9 pour les 18 séries, avec :
#   - la baseline moyenne mobile 3 (meilleure sur Pantin/Bois)
#   - la variante 2 (meilleure sur l'ensemble des séries, MAE 38.1 vs 41.1)
# Rien n'est connu de S9 : on ajoute 18 lignes vides et on ne calcule leurs
# variables qu'à partir de S6, S7, S8 (shift >= 1, testé sans fuite).
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

df = pd.read_csv("data/processed/pesees_propres.csv")[["site", "categorie", "semaine", "collecte_kg"]]

# Lignes S9 : collecte inconnue
s9 = df[df["semaine"] == 8][["site", "categorie"]].assign(semaine=9, collecte_kg=float("nan"))
df = (pd.concat([df, s9]).sort_values(["site", "categorie", "semaine"]).reset_index(drop=True))

g = df.groupby(["site", "categorie"])["collecte_kg"]
for k in (1, 2):
    df[f"lag_{k}"] = g.shift(k)
df["moy_3"]   = g.transform(lambda s: s.shift(1).rolling(3).mean())
df["moy_exp"] = g.transform(lambda s: s.shift(1).expanding().mean())
df["r1"]    = df["lag_1"] / df["moy_3"]
df["r2"]    = df["lag_2"] / df["moy_3"]
df["r_exp"] = df["moy_3"] / df["moy_exp"]

num = ["r1", "r2", "r_exp"]
train = df[(df["semaine"] <= 8)].dropna(subset=num + ["moy_3"])   # S4 à S8 : 90 lignes
futur = df[df["semaine"] == 9].copy()

v2 = Pipeline([("prep", ColumnTransformer([("num", StandardScaler(), num)])),
               ("model", Ridge(alpha=10.0))])           # même alpha qu'à l'évaluation
v2.fit(train[num], train["collecte_kg"] / train["moy_3"] - 1)

futur["baseline_moy3"] = futur["moy_3"]
futur["v2"] = futur["moy_3"] * (1 + v2.predict(futur[num]))

out = futur[["site", "categorie", "baseline_moy3", "v2"]].round(1)
print(f"Lignes d'entraînement V2 : {len(train)}")
print(out.to_string(index=False))
print("\nTotal par site :")
print(out.groupby("site")[["baseline_moy3", "v2"]].sum().round(0))
out.to_csv("previsions_S9.csv", index=False)
