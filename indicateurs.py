


# taux de recyclage
def taux(lignes):
    pourcent = 0.00
    tx_recycled =  total_recycled(lignes)  
    qte_collected = 0
    for line in lignes:
        qte_collected += line["collecte_kg"]
    
        
    pourcent = (tx_recycled/qte_collected)*100  
    
    return  pourcent


#calcul du  total   recycle
def total_recycled(lignes):  
    tt = 0
    for line in lignes:
        tt += line["recycle_kg"]
    
    return tt     

#collecte maximum
def maxi(lignes):
    
    max = 0
    for line in lignes:
       if line["collecte_kg"] > max:
           max = line["collecte_kg"]
           
    return max       