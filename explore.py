
from indicateurs import total_recycled , taux, maxi
from charger import charger_donnees


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

#le_total = total(lignes)
print(f"la quantité totale recyclé est de :{total_recycled(lignes)} Kg")

#le_taux = taux(lignes)
print(f"le taux de recyclage  est de : {taux(lignes):.2f} %")


#le_maxi = maxi(lignes)
print(f"la quantité maximum de collecte est de :{maxi(lignes)} Kg")

print("############### exercice 6 ###################")  

#charger un fichier    
lignes = charger_donnees(open(r"data/pesees.csv"))
#le_total = total(lignes)
print(f"la quantité totale recyclé est de :{total_recycled(lignes)} Kg")

#le_taux = taux(lignes)
print(f"le taux de recyclage  est de : {taux(lignes):.2f} %")


#le_maxi = maxi(lignes)
print(f"la quantité maximum de collecte est de :{maxi(lignes)} Kg")    

print("############### exercice 7 ###################")  

    