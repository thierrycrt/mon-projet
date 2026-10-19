import pandas as pd


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



#df = pd.read_csv("data/brut/pesees.csv")


#le_taux = taux(lignes)
tri_total = df["revendu_kg"] + df["recycle_kg"] + df["dechetterie_kg"]
tx = (df["revendu_kg"] + df["recycle_kg"]).sum() / tri_total.sum()
print(f"le taux de recyclage  est de : {tx:.1f} %")


#le_total = total(lignes)
total_kg = tri_total.sum()
print(f"Total matière : {total_kg:,.0f} kg")




# 3. Verification de la categorie la plus jetee
# (Exemple : categorie avec la plus grande quantité en déchetterie/poubelle)
cat_top = df.groupby("categorie")["dechetterie_kg"].sum().idxmax()
print(f"Catégorie la plus jetée : {cat_top}")





print("############### exercice 2.3 ###################") 
#df = pd.read_csv("data/brut/pesees.csv")

cols_numeriques = ["collecte_kg", "revendu_kg", "recycle_kg", "dechetterie_kg", "attente_kg"]
for col in cols_numeriques:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
print("-------------- VALEURS MANQUANTES APRÈS TYPAGE ------------------")    
 
print(df.isna().sum())  
 
print("-------------- infos APRÈS TYPAGE ------------------")  
        
print(df.info())

print("############### exercice 2.4 ###################")  

# 1. Chargement et typage 

#df = pd.read_csv("data/brut/pesees.csv")
 
print(df.duplicated().sum())
init_lines = len(df)



#df.duplicated(subset=None, keep='first')

strict_duplicated_nb = df.duplicated().sum()
df=df.drop_duplicates(subset=None, keep='first', inplace=True)

job_key = ["site", "semaine", "categorie"]
job_duplicate_job = df.duplicated(subset=job_key).sum()


df=df.drop_duplicates(subset=job_key, keep='last', inplace=True)

print("--- BILAN DÉDOUBLONNAGE ---")

final_line=len(df)
print(f"Lignes initiales : {init_lines}")
print(f"Doublons stricts supprimés : {strict_duplicated_nb}")
print(f"Doublons métier supprimés : {job_duplicate_job}")
print(f"Nombre de lignes après suppression des doublons : {final_line}")


print("############### exercice 2.5 ###################")  


#Detect les valeur manquantes dans df et l'affiche sous forme de somme par colonnes
print(df.isna().sum())

#creation des liste de numeriques pour l'imputation
cols_list = [
    "collecte_kg",
    "revendu_kg",
    "recycle_kg",
    "dechetterie_kg",
    "attente_kg",
]

#parcours de la liste des colonnes pour convertir en numérique et gérer les valeurs manquantes
for col in cols_list:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# init des valeurs manquantes avant imputation        
manquants_avant = df[cols_list].isna().sum().sum()



# Création des colonnes drapeaux (flags) pour tracer les imputations
for col in cols_list:
    df[f"{col}_imputed"] = df[col].isna()  

# Imputation par la médiane croisée site x categorie (ou juste categorie si site indisponible)
for col in cols_list:
    median_Value = df.groupby(['recycle_kg','revendu_kg'])[col].transform('median')
    df[col]=df[col].fillna(median_Value)
    # sécurité si la médiane croisée est NaN, on remplace par la médiane globale de la colonne
    df[col]=df[col].fillna(df[col].median())  # fallback si la médiane croisée est NaN


manquants_apres = df[cols_list].isna().sum().sum()

# -controle ---------------
print(f"Nombre de valeurs manquantes avant imputation : {manquants_avant}") 
print(f"Nombre de valeurs manquantes après imputation : {manquants_apres}")
print(f"Nombre de valeurs manquantes après imputation traité : {manquants_avant-manquants_apres} valeurs imputées") 

print(median_Value)

print("Nombre de valeurs imputés par colonnes:",
      {col: df[f"{col}_imputed"].sum() for col in cols_list},
      )

#print(df)

#print(df['median_Value'])
#print(df.info())

print("############### exercice 2.6 ###################")
# 1. Chargement, typage, dédoublonnage et imputation
df = pd.read_csv("data/brut/pesees.csv")

ls = df["categorie"].value_counts()

#print(ls)
mapping_categorie ={
    "Textiles": "Textiles",
    "Plastique": "Plastique",
    "PlastiqueS": "Plastique",
    "Papier": "Papier", 
    "Verre": "Verre",
    "Métal": "Métal"
}

df["official_cat"] = df["categorie"].map(mapping_categorie)
unknown_cat = df[df["official_cat"].isna()]["categorie"].nunique()

print(f"\nLibellés non reconnus / inconnus : {unknown_cat}")

df["categorie"] = df["official_cat"].fillna(df["categorie"])
df=df.drop(columns=["official_cat"])


before_filter=len(df)
place= ["Montreuil"]
df=df[df["site"].isin(place)]

line_out=before_filter-len(df)

print("\n--- BILAN HARMONISATION & FILTRAGE ---")
print(f"Nombre de catégories officielles uniques : {df['categorie'].nunique()}")
print(f"Lignes hors périmètre retirées : {line_out}")
print(f"Lignes finales conservées : {len(df)}")

print(df)


print("############### exercice 2.7 ###################")



#value_control(df[cols_list])

total_poid= (df["revendu_kg"] + df["recycle_kg"] + df["dechetterie_kg"] + df["attente_kg"])


#print(total_poid)
#print(df["collecte_kg"])

#assert (df["collecte_kg"] == total_poid).all(), ("le poid total ne correspond pas au poid collecté")
#control_poids(total_poid)

# Assertion 1 : Aucun poids négatif
assert (df[cols_list] >= 0).all().all(), (
    "Erreur : Présence de poids négatifs !"
)

# Assertion 2 : Cohérence de la somme (collecte = revendu + recycle + dechetterie + attente)
somme_poids = (
    df["revendu_kg"] + df["recycle_kg"] + df["dechetterie_kg"] + df["attente_kg"]
)


# Assertion 3 : Unicité de la clé métier (site, semaine, categorie)
assert not df.duplicated(subset=["site", "semaine", "categorie"]).any(), (
    "Erreur : Doublons détectés sur la clé métier !"
)

# Assertion 4 : Aucune valeur manquante restante sur les poids
assert df[cols_list].isna().sum().sum() == 0, (
    "Erreur : Des valeurs manquantes persistent !"
)

assert (df["collecte_kg"] == somme_poids).all(), (
    "Erreur : Incohérence dans la somme des poids !"
)

print("✅ Les 4 assertions sont validées avec succès !")


