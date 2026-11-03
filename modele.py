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
print(mae)

# 5. Prévision pour la semaine suivante (semaine 9)
print("Naïve S9 :", serie.iloc[-1])
print("Moyenne mobile S9 :", serie.tail(3).mean())




