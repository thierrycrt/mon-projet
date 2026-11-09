import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CHEMIN = "data/processed/pesees_propres.csv"

# 1. Variables connues au moment de prédire (toujours décalées)
df = pd.read_csv(CHEMIN).sort_values(["site", "categorie", "semaine"])
g = df.groupby(["site", "categorie"])["collecte_kg"]
for k in (1, 2, 3):
    df[f"lag_{k}"] = g.shift(k)
df["moy_3"] = g.transform(lambda s: s.shift(1).rolling(3).mean())
df = df.dropna(subset=["lag_1", "lag_2", "lag_3", "moy_3"]).reset_index(drop=True)

# 2. Découpage : développement S4 à S7, test S8 (jamais utilisé pour choisir)
dev = df[df["semaine"] <= 7].reset_index(drop=True)
test = df[df["semaine"] == 8].reset_index(drop=True)

BRANCHES = {
    "feat-minimal": ["lag_1"],
    "feat-mean": ["moy_3"],
    "chrono_collecte": ["moy_3", "semaine"],
}
CAT = ["site", "categorie"]


def pipeline(num):
    prep = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])
    return Pipeline([("prep", prep), ("model", Ridge(alpha=1.0))])


# 3. Validation temporelle : on entraîne sur les semaines < S, on valide sur S
plis_temporels = [
    (dev.index[dev["semaine"] < s].to_numpy(), dev.index[dev["semaine"] == s].to_numpy())
    for s in (5, 6, 7)
]


def mae_cv(num, cv):
    X = dev[num + CAT]
    return -cross_val_score(pipeline(num), X, dev["collecte_kg"], cv=cv,
                            scoring="neg_mean_absolute_error")


# 4. Choix de la branche : validation temporelle uniquement
scores = {b: mae_cv(num, plis_temporels).mean() for b, num in BRANCHES.items()}
print("MAE temporelle par branche (S5 à S7) :")
for b, v in scores.items():
    print(f"  {b:<18}{v:6.1f} kg")
meilleure = min(scores, key=scores.get)
num = BRANCHES[meilleure]
print(f"\nMeilleure branche : {meilleure} {num}")

# 5. Les trois mesures sur la meilleure branche
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
mae_kfold = mae_cv(num, kfold)
mae_temp = mae_cv(num, plis_temporels)
final = pipeline(num).fit(dev[num + CAT], dev["collecte_kg"])
mae_test = mean_absolute_error(test["collecte_kg"], final.predict(test[num + CAT]))

print(f"\ncross_val_score KFold mélangé : {mae_kfold.mean():.1f} kg (écart-type {mae_kfold.std():.1f})")
print(f"Validation temporelle         : {mae_temp.mean():.1f} kg, par semaine {np.round(mae_temp, 1)}")
print(f"MAE test (S8)                 : {mae_test:.1f} kg")