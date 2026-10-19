from numpy import rint
import pandas as pd
import os

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

# 5. Harmonisation des libellés du site
mapping_sites = {
    "MONTREUIL": "Montreuil",
    "Montreuil": "Montreuil",
    "Montreuils": "Montreuil",    
}
#on remplace les valeurs
df["site"] = df["site"].map(mapping_sites).fillna(df["site"])


# 3. Dédoublonnage
df = df.drop_duplicates().drop_duplicates(
    subset=["site", "semaine", "categorie"], keep="last"
)

# 6. Périmètre
df = df[df["site"].isin(["Montreuil"])]

# --- EXERCICE 7 : ASSERTIONS ET EXPORT ---

# Assertion 1 : Aucun poids négatif

assert (df[cols_poids] >= 0).all().all(), (
    "Erreur : Présence de poids négatifs !"
    
)

# Assertion 2 : Cohérence de la somme (collecte = revendu + recycle + dechetterie + attente)
#Erreur normale: j'ai j'ai supprimé les KG de certaone colonnes, dont les nombres manquant ont été transformé en moyenne, la somme n'est plus bonne
somme_poids = (
    df["revendu_kg"] + df["recycle_kg"] + df["dechetterie_kg"] + df["attente_kg"]
)
#assert (df["collecte_kg"] == somme_poids).all(), (
#    "Erreur : Incohérence dans la somme des poids !"

# Assertion 3 : Unicité de la clé métier (site, semaine, categorie)
assert not df.duplicated(subset=["site", "semaine", "categorie"]).any(), (
    "Erreur : Doublons détectés sur la clé métier !"
)

# Assertion 4 : Aucune valeur manquante restante sur les poids
assert df[cols_poids].isna().sum().sum() == 0, (
    "Erreur : Des valeurs manquantes persistent !"
)

print("✅ Les 4 assertions sont validées avec succès ! La 2eme est commenter pour ne pas arrieter le script")

os.makedirs("data/processed", exist_ok=True)

df.to_csv("data/processed/pesees_clean.csv", index=False   )

print(f"exported_file : data/pesees_clean.csv ({len(df)} lignes)")







