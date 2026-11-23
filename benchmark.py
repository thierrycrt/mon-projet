# Exercice 1 (J8) : quatre lectures de l'erreur, baseline contre meilleur modèle, même test (semaine 8)
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, max_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import DIMENSIONS, construire

data = construire()
y = data["dechetterie_kg"]
test = data["semaine"] == data["semaine"].max()


def metriques(reel, prevu):
    return {
        "mae": mean_absolute_error(reel, prevu),                         # erreur moyenne, en kg
        "rmse": root_mean_squared_error(reel, prevu),                    # punit les grosses erreurs
        "mape": mean_absolute_percentage_error(reel, prevu) * 100,       # en %, fragile sur les petites valeurs
        "pire_erreur": max_error(reel, prevu),                           # la plus grosse erreur, en kg
    }


def ridge(variables):
    preparation = ColumnTransformer([("nombres", StandardScaler(), variables),
                                     ("dimensions", OneHotEncoder(handle_unknown="ignore"), DIMENSIONS)])
    return Pipeline([("preparation", preparation), ("modele", Ridge(alpha=1))])


# La baseline : la moyenne des 3 semaines d'avant, rien à entraîner
print("Baseline mean3     :", {k: round(v, 1) for k, v in metriques(y[test], data.loc[test, "mean3"]).items()})

# Le meilleur modèle d'hier : le pipeline Ridge feat-mean3, appris sur S4 à S7
variables = ["lag1", "mean3"]
X = data[variables + DIMENSIONS]
modele = ridge(variables).fit(X[~test], y[~test])
print("Ridge feat-mean3   :", {k: round(v, 1) for k, v in metriques(y[test], modele.predict(X[test])).items()})