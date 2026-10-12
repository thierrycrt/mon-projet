import csv 

def load_rows(chemin):
    read = csv.DictReader(chemin)
    lignes=[]    
    for raw in read: 
        raw["semaine"]= int(raw["semaine"])
        raw["collecte_kg"]= int(raw["collecte_kg"])
        raw["revendu_kg"]= int(raw["revendu_kg"])
        raw["recycle_kg"]= int(raw["recycle_kg"])
        raw["dechetterie_kg"]= int(raw["dechetterie_kg"])
        raw["attente_kg"]= int(raw["attente_kg"])
        lignes.append(raw)         
         
    return lignes




