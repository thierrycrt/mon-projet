import pandas as pd

df = pd.read_csv("data/pesees.csv")

df.shape

df.info()

df.head()



def control_poids(total_poid):
    print(df["collecte_kg"])
    assert (total_poid == df["collecte_kg"]).all(), ("le poid total ne correspond pas au poid collecté")
    return 


def value_control(col_list):
    assert (col_list >= 0).all().all(),( "valeur inferieure à 0")
    return 



def key_control():
    return





