import pandas as pd

df=pd.read_csv("data/processed/pesees_propres.csv")

def indicateur(d):
    
    """Indicateur principal : on somme d'abord, on divise ensuite."""
    d["numerateur"] = d["revendu_kg"]+ d["recycle_kg"]+ d["attente_kg"]
    d["denominateur"] = d["revendu_kg"]+ d["recycle_kg"]+ d["dechetterie_kg"]+ d.attente_kg
     
    
    return d["numerateur"].sum()/ d["collecte_kg"].sum()



print(f"Lignes : {len(df)}")

print(f"Indicateur principal : {indicateur(df):.1%} ")

