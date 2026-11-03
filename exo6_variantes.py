print("#-------------------------- Jours 6 exo 6 -------------------------")
# Objectif : battre la moyenne mobile 3 (MAE 16 kg sur Pantin/Bois, S6 à S8)
# avec 3 variantes de variables d'entrée.
#
# Hypothèses [TO CONFIRM] :
#   H1. La prévision de la semaine S est faite en fin de semaine S-1,
#       quand TOUTES les pesées de S-1 (tous sites, toutes catégories) sont saisies.
#   H2. Aucune semaine manquante dans une série (sinon shift(k) ≠ "k semaines avant").
#   H3. Pas de semaine à 0 kg (les variantes 2 et 3 divisent par moy_3).
#
# Protocole : walk-forward. Pour prédire S, on entraîne sur les semaines < S,
# puis on prédit S. Répété pour S = 6, 7, 8 → 3 prévisions comparables à la baseline.
# Les hyperparamètres (alpha) sont fixés AVANT de regarder le test :
# les régler sur S6-S8 serait une fuite.

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

df = (pd.read_csv("data/processed/pesees_propres.csv")
        .sort_values(["site", "categorie", "semaine"])
        .reset_index(drop=True))

# ── Variables communes : uniquement le passé de la série (shift >= 1) ──
g = df.groupby(["site", "categorie"])["collecte_kg"]
for k in (1, 2, 3):
    df[f"lag_{k}"] = g.shift(k)
df["moy_3"]   = g.transform(lambda s: s.shift(1).rolling(3).mean())
df["moy_exp"] = g.transform(lambda s: s.shift(1).expanding().mean())

SEMAINES_TEST = [6, 7, 8]


def walk_forward(data, num, cible, reconstruire, alpha, cat=()):
    """Entraîne sur semaine < S, prédit S, pour chaque S de SEMAINES_TEST."""
    cat = list(cat)
    data = data.dropna(subset=num + ["moy_3"])
    resultats = []
    for s in SEMAINES_TEST:
        tr = data[data["semaine"] < s]
        te = data[data["semaine"] == s]
        blocs = [("num", StandardScaler(), num)]
        if cat:
            blocs.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
        pipe = Pipeline([("prep", ColumnTransformer(blocs)),
                         ("model", Ridge(alpha=alpha))])
        pipe.fit(tr[num + cat], cible(tr))
        p = reconstruire(te, pipe.predict(te[num + cat]))
        resultats.append(te.assign(pred=p))
    return pd.concat(resultats)


def evaluer(nom, res):
    pb = res[(res["site"] == "Pantin") & (res["categorie"] == "Bois")]
    print(f"{nom:<26}"
          f"{mean_absolute_error(pb['collecte_kg'], pb['pred']):>16.1f}"
          f"{mean_absolute_error(res['collecte_kg'], res['pred']):>18.1f}")


print(f"{'Modèle':<26}{'MAE Pantin/Bois':>16}{'MAE toutes séries':>18}")

# ── Baseline (contrôle : doit redonner 16.0 kg sur Pantin/Bois) ──
base = df[df["semaine"].isin(SEMAINES_TEST)].dropna(subset=["moy_3"])
evaluer("Baseline moy. mobile 3", base.assign(pred=base["moy_3"]))


# ═════════════════════════════════════════════════════════════════════════════
# VARIANTE 1 — Niveau + tendance de la série (en kg)
#   lag_1  : collecte de S-1                     → pesée terminée en fin de S-1
#   moy_3  : moyenne de S-1, S-2, S-3            → idem, tout est antérieur à S
#   pente  : (lag_1 - lag_3) / 2, kg par semaine → calculée à partir de lags passés
#   ecart  : lag_1 - moy_3 (S-1 au-dessus ou en dessous de son niveau récent)
#   Pas de one-hot site/catégorie : avec 36 à 72 lignes d'entraînement,
#   18 colonnes indicatrices feraient sur-apprendre (cf. exo 4 et 5).
# ═════════════════════════════════════════════════════════════════════════════
df["pente"] = (df["lag_1"] - df["lag_3"]) / 2
df["ecart"] = df["lag_1"] - df["moy_3"]

v1 = walk_forward(
    df, num=["lag_1", "moy_3", "pente", "ecart"],
    cible=lambda d: d["collecte_kg"],
    reconstruire=lambda d, p: p,
    alpha=1.0,
)
evaluer("V1 niveau + tendance", v1)


# ═════════════════════════════════════════════════════════════════════════════
# VARIANTE 2 — Corriger la baseline au lieu de la remplacer (ratios)
#   Cible : y / moy_3 - 1 = écart relatif à la baseline.
#   Prévision = moy_3 × (1 + correction). Avec un alpha fort, la correction
#   tend vers 0 et le modèle retombe sur la baseline : il ne peut pas faire
#   beaucoup pire qu'elle.
#   Les ratios mettent toutes les séries à la même échelle : le modèle apprend
#   une dynamique commune aux 18 séries au lieu d'apprendre leurs niveaux.
#   r1    : lag_1 / moy_3       → S-1 et S-1..S-3, connus en fin de S-1
#   r2    : lag_2 / moy_3       → S-2, connu
#   r_exp : moy_3 / moy_exp     → niveau récent vs tout l'historique avant S
#   (r3 écarté : r1 + r2 + r3 = 3, il est redondant)
# ═════════════════════════════════════════════════════════════════════════════
df["r1"]    = df["lag_1"] / df["moy_3"]
df["r2"]    = df["lag_2"] / df["moy_3"]
df["r_exp"] = df["moy_3"] / df["moy_exp"]

cible_ratio = lambda d: d["collecte_kg"] / d["moy_3"] - 1
reconstruire_ratio = lambda d, p: d["moy_3"].values * (1 + p)

v2 = walk_forward(
    df, num=["r1", "r2", "r_exp"],
    cible=cible_ratio, reconstruire=reconstruire_ratio,
    alpha=10.0,
)
evaluer("V2 correction relative", v2)


# ═════════════════════════════════════════════════════════════════════════════
# VARIANTE 3 — V2 + signal collectif (site et catégorie en S-1)
#   Idée : si tout le site Pantin a baissé en S-1, ou si le bois a baissé sur
#   tous les sites, c'est une information que la série seule ne voit pas.
#   site_r1 : total du site en S-1 / moyenne du total du site sur S-1..S-3
#   cat_r1  : total de la catégorie (tous sites) en S-1 / moyenne S-1..S-3
#   Disponibles UNIQUEMENT sous H1 (toutes les pesées S-1 saisies en fin de S-1).
#   Le total de la semaine S n'est jamais utilisé : il est décalé de 1 semaine
#   avant tout calcul.
# ═════════════════════════════════════════════════════════════════════════════
df["tot_site"] = df.groupby(["site", "semaine"])["collecte_kg"].transform("sum")
df["tot_cat"]  = df.groupby(["categorie", "semaine"])["collecte_kg"].transform("sum")

for tot in ("tot_site", "tot_cat"):
    gt = df.groupby(["site", "categorie"])[tot]
    df[f"{tot}_lag1"] = gt.shift(1)
    df[f"{tot}_moy3"] = gt.transform(lambda s: s.shift(1).rolling(3).mean())

df["site_r1"] = df["tot_site_lag1"] / df["tot_site_moy3"]
df["cat_r1"]  = df["tot_cat_lag1"]  / df["tot_cat_moy3"]

v3 = walk_forward(
    df, num=["r1", "r2", "r_exp", "site_r1", "cat_r1"],
    cible=cible_ratio, reconstruire=reconstruire_ratio,
    alpha=10.0,
)
evaluer("V3 + signal collectif", v3)

# ─────────────────────────────────────────────────────────────────────────────
# VERDICT (walk-forward S6-S8)
#                              Pantin/Bois   Toutes séries (54 prév.)
#   Baseline moy. mobile 3        16.0           41.1
#   V1 niveau + tendance          32.6           41.1   → MOINS BIEN / égal
#   V2 correction relative        27.9           38.1   → MOINS BIEN / MIEUX (-3.0)
#   V3 + signal collectif         23.3           39.0   → MOINS BIEN / MIEUX (-2.1)
#
# Hypothèses vérifiées : 8 semaines pour chaque série, aucun 0, aucune
#   collecte imputée.
#
# Objectif 16 kg NON atteint. Aucune variante ne bat la baseline sur Pantin/Bois.
# V2 est la seule piste réelle : -3 kg sur l'ensemble des séries.
#
# Pourquoi Pantin/Bois résiste : la série est plate et bruitée (413 à 462 kg
#   de S3 à S8). Une semaine haute (S5 : 459) n'annonce pas la suivante.
#   Les modèles apprennent sur les 18 séries des dynamiques qui ne
#   s'appliquent pas à celle-ci.
# Piège évité : la moyenne de tout l'historique (moy_exp) ferait 10.3 kg sur
#   Pantin/Bois, mais 50.6 kg toutes séries. La retenir parce qu'elle gagne
#   sur ces 3 semaines serait choisir le modèle sur le test.
# Essai : retirer Bagnolet S1 (collectes à 42 % du niveau habituel,
#   semaine probablement partielle) n'améliore rien (V2 : 32.8 / 39.4).
# Aucun réglage supplémentaire n'a été fait : régler sur S6-S8 = fuite.
# ─────────────────────────────────────────────────────────────────────────────
