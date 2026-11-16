import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CHEMIN = "data/processed/pesees_propres.csv"
NUM = ["moy_3", "semaine"]   # variables de la branche chrono_collecte
CAT = ["site", "categorie"]

# 1. Variables connues au moment de prédire (décalées)
df = pd.read_csv(CHEMIN).sort_values(["site", "categorie", "semaine"])
g = df.groupby(CAT)["collecte_kg"]
df["moy_3"] = g.transform(lambda s: s.shift(1).rolling(3).mean())
df = df.dropna(subset=["moy_3"]).reset_index(drop=True)
X, y = df[NUM + CAT], df["collecte_kg"]

# 2. Pipeline : encodage et modèle sont sauvegardés ensemble
pipe = Pipeline([
    ("prep", ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
    ])),
    ("model", Ridge(alpha=1.0)),
])

# 3. Contrôle : apprentissage S4 à S7, test S8 (on doit retrouver 39,8 kg)
train, test = df["semaine"] <= 7, df["semaine"] == 8
pipe.fit(X[train], y[train])
print(f"MAE test S8 : {mean_absolute_error(y[test], pipe.predict(X[test])):.1f} kg")

# 4. Modèle final : réentraîné sur toutes les semaines disponibles (S4 à S8)
pipe.fit(X, y)
joblib.dump(pipe, "modele.joblib")

# 5. Vérification : le modèle rechargé donne exactement les mêmes prévisions
modele = joblib.load("modele.joblib")
assert np.array_equal(pipe.predict(X), modele.predict(X))

# 6. Utilisation : prévision de la semaine 9
historique = pd.read_csv(CHEMIN)
s9 = (historique[historique["semaine"].between(6, 8)]
      .groupby(CAT, as_index=False)["collecte_kg"].mean()
      .rename(columns={"collecte_kg": "moy_3"})
      .assign(semaine=9))
s9["prevision_kg"] = modele.predict(s9[NUM + CAT]).round(0).astype(int)
print(s9.pivot(index="categorie", columns="site", values="prevision_kg"))
print("Total S9 :", s9["prevision_kg"].sum(), "kg")