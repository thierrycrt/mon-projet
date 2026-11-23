# Exercice 2 (J8) : l'alerte de débordement (UC-02) en classification — précision, rappel, seuil
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import DIMENSIONS, charger

SEUIL_KG = 50                                      # la zone d'attente déborde à partir de 50 kg (règle de Nadia)

df = charger()
g = df.groupby(["site", "categorie"])
df["attente_lag1"] = g["attente_kg"].shift(1)                                         # attente de S-1
df["attente_mean3"] = g["attente_kg"].transform(lambda s: s.shift(1).rolling(3).mean())
df["collecte_lag1"] = g["collecte_kg"].shift(1)
df["deborde"] = (df["attente_kg"] >= SEUIL_KG).astype(int)                            # la cible : 1 = débordement
data = df.dropna(subset=["attente_mean3", "collecte_lag1"])                           # semaines 4 à 8

VARIABLES = ["attente_lag1", "attente_mean3", "collecte_lag1"]
X, y = data[VARIABLES + DIMENSIONS], data["deborde"]
test = data["semaine"] >= 7                        # 2 semaines de test : assez de débordements pour compter

modele = Pipeline([
    ("preparation", ColumnTransformer([("nombres", StandardScaler(), VARIABLES),
                                       ("dimensions", OneHotEncoder(handle_unknown="ignore"), DIMENSIONS)])),
    ("modele", LogisticRegression(max_iter=1000)),
]).fit(X[~test], y[~test])
proba = modele.predict_proba(X[test])[:, 1]        # probabilité de débordement, entre 0 et 1

print(f"Débordements réels dans le test : {y[test].sum()} sur {test.sum()} lignes")
print(f"Toujours dire « pas de débordement » : exactitude {accuracy_score(y[test], [0] * test.sum()):.0%}, rappel 0 %")
regle = (X.loc[test, "attente_lag1"] >= SEUIL_KG).astype(int)                        # la règle d'aujourd'hui
for nom, alerte in [("Règle : déborde si S-1 a débordé", regle)] + \
                   [(f"Modèle, seuil {s}", (proba >= s).astype(int)) for s in [0.7, 0.5, 0.1, 0.05]]:
    tn, fp, fn, tp = confusion_matrix(y[test], alerte, labels=[0, 1]).ravel()
    print(f"{nom:35s} attrapés {tp} · ratés {fn} · fausses alertes {fp} · "
          f"précision {precision_score(y[test], alerte, zero_division=0):.0%} · rappel {recall_score(y[test], alerte):.0%}")
    
    
# Pourquoi une exactitude de 75 % ne veut rien dire ici ?
# L'exactitude de 75% ne veut rien dire car un score doit etre nettement meilleur pour signifier quelque chose,
# et surtout il ne detecte aucun des 25% (9) de debordement reels (rappel 0%).

# Que se passe-t-il quand on baisse le seuil, de 0,7 à 0,05 ?
# Plus on baisse , plus on a de debordements detectes, les rappel  augmentent, mais aussi plus de fausses alertes la precision diminue.
# mais plus on a un resultat significatif.

# Qu’est-ce qui coûte le plus cher sur le terrain, un débordement raté ou une fausse alerte ? 
# sur le terrain, un faux positif a l'air plus facile à gerer qu'un debordement reel
# Quel seuil choisissez-vous ?
# Dans se cas precis, il vaut mieux connaitre les debordements detectes, donc on choisit le seuil de "attrapés 9, raté 0 et fausse alerte 1"
# seuil 0.1

# Le modèle fait-il mieux que la règle d’aujourd’hui ?
# oui, le modele fait mieux que la regle d'aujourd'hui.
#il detecte plus de débordements et a moins de fauses alertes .