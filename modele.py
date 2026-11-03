import pandas as pd
import numpy as np

print("#-------------------------- Jours 6 exo 2-------------------------")
#basé sur "data/processed/pesees_propres.csv"  144 lignes, 8 semaines, 3 sites, 6 catégories
df = pd.read_csv("data/processed/pesees_propres.csv")

# 1) Filtre Pantin + Bois (majuscules comme dans le fichier)
y = (df[(df["site"] == "Pantin") & (df["categorie"] == "Bois")]
     .sort_values("semaine")
     .set_index("semaine")["collecte_kg"])

# 2) Baselines calculées sur toute la série (elles ont besoin de l'historique)
baseline_naive = y.shift(1)                     # semaine t = valeur de t-1
baseline_mm3   = y.shift(1).rolling(3).mean()   # semaine t = moyenne de t-3, t-2, t-1


print("\n=== Scores sur ces 3 dernieres semaines du site Pantin en collecte de bois ===")
# 3) On ne garde que les 3 dernières semaines
resultat = pd.DataFrame({
    "reel_kg": y,
    "baseline_naive": baseline_naive,
    "baseline_mm3": baseline_mm3,
}).tail(3).round(1)
print(resultat)


# 4) Scores sur ces 3 semaines de pantin en term
for nom in ["baseline_naive", "baseline_mm3"]:
    err = resultat["reel_kg"] - resultat[nom]
    print(f"{nom:15s} MAE = {err.abs().mean():.1f} kg | RMSE = {np.sqrt((err**2).mean()):.1f} kg")





