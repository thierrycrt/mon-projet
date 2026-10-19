import pandas as pd

df = pd.read_csv("data/brut/pesees.csv")

df.shape

df.info()

df.head()

# 1. Chargement du fichier brut
df = pd.read_csv("data/brut/pesees.csv")

# 2. Typage des colonnes
cols_poids = [
    "collecte_kg",
    "revendu_kg",
    "recycle_kg",
    "dechetterie_kg",
    "attente_kg",
]
for col in cols_poids:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["semaine"] = pd.to_numeric(df["semaine"], errors="coerce")

# 3. Dédoublonnage
df = df.drop_duplicates().drop_duplicates(
    subset=["site", "semaine", "categorie"], keep="last"
)

# 4. Imputation des valeurs manquantes et drapeaux
for col in cols_poids:
    df[f"{col}_imputed"] = df[col].isna()
    mediane_groupe = df.groupby(["site", "categorie"])[col].transform("median")
    df[col] = df[col].fillna(mediane_groupe).fillna(df[col].median())

# 5. Harmonisation des libellés
mapping_categories = {
    "Textiles": "Textile",
    "Textile": "Textile",
    "Papier": "Papier",
    "Plastique": "Plastique",
    "Verre": "Verre",
    "Métal": "Métal",
    "Bois": "Bois",
}

df["categorie"] = df["categorie"].map(mapping_categories).fillna(df["categorie"])

# 6. Périmètre
df = df[df["site"].isin(["Montreuil"])]

# --- EXERCICE 7 : ASSERTIONS ET EXPORT ---

# Assertion 1 : Aucun poids négatif
#Erreur normale: j'ai j'ai supprimé les KG de certaone colonnes, dont les nombres manquant ont été transformé en motenne, la somme n'est plus bonne
assert (df[cols_poids] >= 0).all().all(), (
    "Erreur : Présence de poids négatifs !"
)









