
from indicateurs import total_recycled , taux, maxi, tx, max, total
from load import load_rows
from nettoyage import df
import pandas as pd


print("############### exercice 3 ###################")  
lignes = [
    {
        "site": "Bordeaux",
        "semaine": 12,
        "categorie": "Textiles",
        "collecte_kg": 1206,
        "revendu_kg": 249,
        "recycle_kg": 221,
        "dechetterie_kg": 635,
        "attente_kg": 101
    },
    {
        "site": "Bordeaux",
        "semaine": 12,
        "categorie": "metaux",
        "collecte_kg": 1365,
        "revendu_kg": 400,
        "recycle_kg": 200,
        "dechetterie_kg": 600,
        "attente_kg": 165
        
    },
    {
        "site": "Bordeaux",
        "semaine": 12,
        "categorie": "Papiers",
        "collecte_kg": 1300,
        "revendu_kg": 300,
        "recycle_kg": 300,
        "dechetterie_kg": 600,
        "attente_kg": 100
        },
    {
        "site": "Bordeaux",
        "semaine": 12,
        "categorie": "bois",
        "collecte_kg": 1400,
        "revendu_kg": 400,
        "recycle_kg": 200,
        "dechetterie_kg": 600,
        "attente_kg": 200        
        
    },
    {
        "site": "Bordeaux",
        "semaine": 12,
        "categorie": "verre",
        "collecte_kg": 1000,
        "revendu_kg": 200,
        "recycle_kg": 200,
        "dechetterie_kg": 500,
        "attente_kg": 100
        
        }   
    
]


print(len(lignes))
print(lignes[0]["site"])

#for elements in lignes:
    #print(elements["site"])
    
print("############### exercice 4 ###################")   
nb=0   
for ligne in lignes:
    if ligne["site"]=="Bordeaux" and ligne["semaine"]==12:
       nb += 1 
kg = 0
site=""
for ligne in lignes:
    if ligne["site"]=="Bordeaux" and ligne["semaine"]==12:
      site = ligne["site"]
      kg += ligne["collecte_kg"] + ligne["revendu_kg"]
      
      
print(f"{site}, semaine: {ligne["semaine"]} :{nb} enregistrement ")
print(f"{site}, semaine: {ligne["semaine"]} :{kg} kg triés ")



print("############### exercice 5 ###################")  
df = pd.read_csv("data/pesees.csv")

#le_total = total(lignes)
print(f"la quantité totale recyclé est de :{total(df)} Kg")

#le_taux = taux(lignes)
print(f"le taux de recyclage  est de : {tx(df):.2f} %")


#le_maxi = maxi(lignes)
print(f"la quantité maximum de collecte est de :{max(df)} Kg")



print("############### exercice 6 ###################")  

#charger un fichier    
lignes = load_rows(open(r"data/pesees.csv"))
#le_total = total(lignes)
print(f"la quantité totale recyclé est de :{total_recycled(lignes)} Kg")

#le_taux = taux(lignes)
print(f"le taux de recyclage  est de : {taux(lignes):.2f} %")


#le_maxi = maxi(lignes)
print(f"la quantité maximum de collecte est de :{maxi(lignes)} Kg")    




print("############### pandas exercice 2.2 ###################")  
df = pd.read_csv("data/pesees.csv")

#le_total = total(lignes)
print(f"la quantité totale recyclé est de :{total(df)} Kg")

#le_taux = taux(lignes)
print(f"le taux de recyclage  est de : {tx(df):.2f} %")


#le_maxi = maxi(lignes)
print(f"la quantité maximum de collecte est de :{max(df)} Kg")

print("############### exercice 2.3 ###################") 
df = pd.read_csv("data/brut/pesees.csv")

cols_numeriques = ["revendu_kg", "recycle_kg", "dechetterie_kg"]
for col in cols_numeriques:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
print("-------------- VALEURS MANQUANTES APRÈS TYPAGE ------------------")    
 
print(df.isna().sum())  
 
print("-------------- infos APRÈS TYPAGE ------------------")  
        
print(df.info())

print("############### exercice 2.4 ###################")   
 
print(df.duplicated().sum())
init_lines = len(df)


        
        

df.duplicated(subset=None, keep='first')
strict_duplicates_nb = df.duplicated().sum()
df.drop_duplicates(subset=None, keep='first', inplace=True)
job_duplicate_job = df.duplicated(subset=None, keep='last' ).sum()
print(df.duplicated().sum())

print("--- BILAN DÉDOUBLONNAGE ---")

final_line=len(df)
print(f"Lignes initiales : {init_lines}")
print(f"Doublons stricts supprimés : {strict_duplicates_nb}")
print(f"Doublons métier supprimés : {job_duplicate_job}")
print(f"Nombre de lignes après suppression des doublons : {final_line}")


print("############### exercice 2.5 ###################")  

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

ls = df["categorie"].value_counts()
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




        

