import pandas as pd
import os


print("############### pandas exercice 2.1 ###################")


# Chargement du fichier CSV
df = pd.read_csv("data/brut/pesees.csv")

# 1. Nombre de lignes et de colonnes
print("--- SHAPE ---")
print(df.shape)

# 2. Types de colonnes et valeurs manquantes (trous)
print("\n--- INFO ---")
print(df.info())

# 3. Aperçu des premières lignes
print("\n--- HEAD ---")
print(df.head())


print("############### pandas exercice 2.2 ###################")  



df = pd.read_csv("data/brut/pesees.csv")


# 1. Verification du taux de revalorisation / tri
# (Exemple : somme (revendu + recycle) / somme totale du tri)
tri_total = df["revendu_kg"] + df["recycle_kg"] + df["dechetterie_kg"]
taux = (df["revendu_kg"] + df["recycle_kg"]).sum() / tri_total.sum()
print(f"Taux de revalorisation : {taux:.1%}")

# 2. Verification du tonnage / total des poids
# (Exemple : somme totale de la matière triée/traitée)
total_kg = tri_total.sum()
print(f"Total matière : {total_kg:,.0f} kg")

# 3. Verification de la categorie la plus jetee
# (Exemple : categorie avec la plus grande quantité en déchetterie/poubelle)
cat_top = df.groupby("categorie")["dechetterie_kg"].sum().idxmax()
print(f"Catégorie la plus jetée : {cat_top}")





print("############### exercice 2.3 ###################") 
df = pd.read_csv("data/brut/pesees.csv")

print("--- NOMS DES COLONNES DISPONIBLES ---")
print(df.columns.tolist())

cols_numeriques = ["revendu_kg", "recycle_kg", "dechetterie_kg"]

# Conversion des nombres
for col in cols_numeriques:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Vérifications
print("\n--- VALEURS MANQUANTES APRÈS TYPAGE ---")
print(df.isna().sum())

print("\n--- INFO APRÈS TYPAGE ---")
print(df.info())

print("############### exercice 2.4 ###################") 
df = pd.read_csv("data/brut/pesees.csv")


# 2. Typage des colonnes
cols_numeriques = ["collecte_kg", "revendu_kg", "recycle_kg", "dechetterie_kg", "attente_kg"]
for col in cols_numeriques:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["semaine"] = pd.to_numeric(df["semaine"], errors="coerce")

lignes_initiales = len(df)

# --- EXERCICE 4 : DÉDOUBLONNAGE ---

# A. Compter et supprimer les doublons stricts (lignes 100 % identiques)
nb_doublons_stricts = df.duplicated().sum()
df = df.drop_duplicates()

# B. Compter et trancher la règle pour les doublons métier
# Clé métier : site + semaine + categorie
cle_metier = ["site", "semaine", "categorie"]
nb_doublons_metier = df.duplicated(subset=cle_metier).sum()

# Règle métier appliquée : en cas de doublon sur la clé métier,
# on conserve la dernière entrée enregistrée (keep="last")
df = df.drop_duplicates(subset=cle_metier, keep="last")

lignes_finales = len(df)
total_supprime = lignes_initiales - lignes_finales

# --- VERIFICATION ---
print("--- BILAN DÉDOUBLONNAGE ---")
print(f"Lignes initiales : {lignes_initiales}")
print(f"Doublons stricts supprimés : {nb_doublons_stricts}")
print(f"Doublons métier supprimés (clé site/semaine/catégorie) : {nb_doublons_metier}")
print(f"Lignes restantes : {lignes_finales}")


print("############### exercice 2.5 ###################")  


# 1. Chargement et typage
df = pd.read_csv("data/brut/pesees.csv")

cols_numeriques = [
    "collecte_kg",
    "revendu_kg",
    "recycle_kg",
    "dechetterie_kg",
    "attente_kg",
]
for col in cols_numeriques:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["semaine"] = pd.to_numeric(df["semaine"], errors="coerce")

# 2. Dédoublonnage
cle_metier = ["site", "semaine", "categorie"]
df = df.drop_duplicates().drop_duplicates(subset=cle_metier, keep="last")

# --- EXERCICE 5 : TRAITEMENT DES VALEURS MANQUANTES ---

# Carte initiale des trous
manquants_avant = df[cols_numeriques].isna().sum().sum()

# Création des colonnes drapeaux (flags) pour tracer les imputations
for col in cols_numeriques:
    df[f"{col}_imputed"] = df[col].isna()

# Imputation par la médiane croisée site x categorie (ou juste categorie si site indisponible)
for col in cols_numeriques:
    mediane_groupe = df.groupby(["site", "categorie"])[col].transform("median")
    df[col] = df[col].fillna(mediane_groupe)
    # Sécurité si des groupes entiers étaient vides : imputation par la médiane globale
    df[col] = df[col].fillna(df[col].median())

manquants_apres = df[cols_numeriques].isna().sum().sum()

# --- VERIFICATION ---
print("--- BILAN TRAITEMENT DES MANQUANTS ---")
print(f"Valeurs manquantes traitées : {manquants_avant - manquants_apres}")
print(f"Reste de valeurs manquantes : {manquants_apres}")
print(
    "Nombre de valeurs imputées par colonne :",
    {col: df[f"{col}_imputed"].sum() for col in cols_numeriques},
)









print("############### exercice 2.6 ###################")



# 1. Chargement, typage, dédoublonnage et imputation
df = pd.read_csv("data/brut/pesees.csv")

cols_numeriques = [
    "collecte_kg",
    "revendu_kg",
    "recycle_kg",
    "dechetterie_kg",
    "attente_kg",
]
for col in cols_numeriques:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["semaine"] = pd.to_numeric(df["semaine"], errors="coerce")

df = df.drop_duplicates().drop_duplicates(
    subset=["site", "semaine", "categorie"], keep="last"
)

# Imputation des valeurs manquantes si présentés
for col in cols_numeriques:
    df[f"{col}_imputed"] = df[col].isna()
    mediane_groupe = df.groupby(["site", "categorie"])[col].transform("median")
    df[col] = df[col].fillna(mediane_groupe).fillna(df[col].median())

# --- EXERCICE 6 : HARMONISATION DES LIBELLÉS ET PÉRIMÈTRE ---

print("--- LIBELLÉS BRUTS EN ENTRÉE ---")
print(df["categorie"].value_counts())

# Normalisation explicite sur les données du CSV
mapping_categories = {
    "Textiles": "Textile",
    "Textile": "Textile",
    "Papier": "Papier",
    "Plastique": "Plastique",
    "Verre": "Verre",
    "Métal": "Métal",
    "Bois": "Bois",
}

# Application du mapping
df["categorie_officielle"] = df["categorie"].map(mapping_categories)
inconnus = df[df["categorie_officielle"].isna()]["categorie"].unique()

print(f"\nLibellés non reconnus / inconnus : {inconnus}")

df["categorie"] = df["categorie_officielle"].fillna(df["categorie"])
df = df.drop(columns=["categorie_officielle"])

# Filtrage du périmètre sur le site pilote Montreuil
lignes_avant_filtre = len(df)
sites_officiels = ["Montreuil"]
df = df[df["site"].isin(sites_officiels)]

lignes_retirees = lignes_avant_filtre - len(df)

# --- BILAN ---
print("\n--- BILAN HARMONISATION & FILTRAGE ---")
print(f"Nombre de catégories officielles uniques : {df['categorie'].nunique()}")
print(f"Lignes hors périmètre retirées : {lignes_retirees}")
print(f"Lignes finales conservées : {len(df)}")


print("############### exercice 2.7 ###################")

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
assert (df[cols_poids] >= 0).all().all(), (
    "Erreur : Présence de poids négatifs !"
)

# Assertion 2 : Cohérence de la somme (collecte = revendu + recycle + dechetterie + attente)
somme_poids = (
    df["revendu_kg"] + df["recycle_kg"] + df["dechetterie_kg"] + df["attente_kg"]
)
assert (df["collecte_kg"] == somme_poids).all(), (
    "Erreur : Incohérence dans la somme des poids !"
)

# Assertion 3 : Unicité de la clé métier (site, semaine, categorie)
assert not df.duplicated(subset=["site", "semaine", "categorie"]).any(), (
    "Erreur : Doublons détectés sur la clé métier !"
)

# Assertion 4 : Aucune valeur manquante restante sur les poids
assert df[cols_poids].isna().sum().sum() == 0, (
    "Erreur : Des valeurs manquantes persistent !"
)

print("✅ Les 4 assertions sont validées avec succès !")

# Export du fichier propre
os.makedirs("data", exist_ok=True)
df.to_csv("data/pesees_propres.csv", index=False)

print(f"📁 Fichier exporté : data/pesees_propres.csv ({len(df)} lignes)")
