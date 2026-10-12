


# taux de recyclage
def taux(lignes):
    pourcent = 0.00
    tx_recycled =  total_recycled(lignes)  
    qte_collected = 0
    for line in lignes:
        qte_collected += line["collecte_kg"]
    
        
    pourcent = (tx_recycled/qte_collected)*100  
    
    return  pourcent

def tx(df):
    
    #tx_recycled   = df["recycled_kg"].sum()
    #total_collected =df["collected_kg"].sum()
    percent=(df["recycle_kg"].sum()/df["collecte_kg"].sum())*100
    
    return percent

#calcul du  total   recycle
def total_recycled(lignes):  
    tt = 0
    for line in lignes:
        tt += line["recycle_kg"]
    
    return tt   

def total(df):
    return df["recycle_kg"].sum()  

#collecte maximum
def maxi(lignes):
    
    max = 0
    for line in lignes:
       if line["collecte_kg"] > max:
           max = line["collecte_kg"]
           
    return max  

def max(df):
    return df["collecte_kg"].max()  
     