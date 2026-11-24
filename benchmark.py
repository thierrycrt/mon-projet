# J8 : benchmark tracé dans MLflow (baseline + modèles de J6 et J7), sur le même test : semaine 8
import subprocess
import time

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, max_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("my-experiment")

from features import DIMENSIONS, construire

data = construire()
y = data["dechetterie_kg"]
test = data["semaine"] == data["semaine"].max()
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def metriques(reel, prevu):
    return {
        "mae": mean_absolute_error(reel, prevu),
        "rmse": root_mean_squared_error(reel, prevu),
        "mape": mean_absolute_percentage_error(reel, prevu) * 100,
        "pire_erreur": max_error(reel, prevu),
    }


def ridge(variables):
    prep = ColumnTransformer([("nombres", StandardScaler(), variables),
                              ("dimensions", OneHotEncoder(handle_unknown="ignore"), DIMENSIONS)])
    return Pipeline([("preparation", prep), ("modele", Ridge(alpha=1))])


def dummies(variables):
    return pd.get_dummies(data[variables + DIMENSIONS], columns=DIMENSIONS)


ESSAIS = {
    "linear (J6)": (LinearRegression(), dummies(["lag1", "mean3"])),
    "tree free (J6)": (DecisionTreeRegressor(random_state=0), dummies(["lag1", "mean3"])),
    "tree depth 3 (J6)": (DecisionTreeRegressor(max_depth=3, random_state=0), dummies(["lag1", "mean3"])),
    "ridge feat-lag1 (J7)": (ridge(["lag1"]), data[["lag1"] + DIMENSIONS]),
    "ridge feat-mean3 (J7)": (ridge(["lag1", "mean3"]), data[["lag1", "mean3"] + DIMENSIONS]),
    "ridge feat-collecte (J7)": (ridge(["lag1", "mean3", "collecte_lag1"]), data[["lag1", "mean3", "collecte_lag1"] + DIMENSIONS]),
}

mlflow.set_experiment("J8 - benchmark")
lignes = []

with mlflow.start_run(run_name="baseline mean3"):            # la baseline se logge aussi
    m = metriques(y[test], data.loc[test, "mean3"])
    mlflow.log_params({"variables": "mean3", "git_commit": commit})
    mlflow.log_metrics({**m, "duree_fit_s": 0.0})
    lignes.append({"essai": "baseline mean3", **m, "duree_fit_s": 0.0})

for nom, (modele, X) in ESSAIS.items():
    with mlflow.start_run(run_name=nom):
        debut = time.perf_counter()
        modele.fit(X[~test], y[~test])
        duree = time.perf_counter() - debut
        m = metriques(y[test], modele.predict(X[test]))
        mlflow.log_params({"variables": ", ".join(c for c in X.columns if not c.startswith(("site", "categorie"))), "git_commit": commit})
        mlflow.log_metrics({**m, "duree_fit_s": duree})
        lignes.append({"essai": nom, **m, "duree_fit_s": duree})

tableau = pd.DataFrame(lignes).sort_values("mae")
print(tableau.round({"mae": 1, "rmse": 1, "mape": 1, "pire_erreur": 0, "duree_fit_s": 3}).to_string(index=False))