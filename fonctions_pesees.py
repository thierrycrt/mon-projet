"""Fonctions de calcul pour pesees.csv (Courtieu - Bloc 2, 2026).

Python standard uniquement. Les fonctions travaillent sur la liste de
dictionnaires renvoyee par load_rows() dans rapport_pesees.py.
Usage : python fonctions_pesees.py pesees.csv
"""
import sys


def total_recycle(rows):
    """Retourne le total recycle (somme de recycle_kg) sur tout le fichier.

    Parametre :
        rows : liste de dictionnaires renvoyee par load_rows().
    Retour :
        un entier, en kg.
    Exemple :
        total_recycle(rows)  ->  somme de la colonne recycle_kg
    """
    return sum(row["recycle_kg"] for row in rows)


def taux_recyclage(rows):
    """Retourne le taux de recyclage en pourcentage sur tout le fichier.

    Regle : somme de recycle_kg / somme de collecte_kg * 100.
    [TO CONFIRM] Votre message ecrit "/100", ce qui donnerait une fraction
    tres petite (0.005 au lieu de 0.5 %). J'ai applique "* 100" pour obtenir
    un pourcentage. Si vous voulez vraiment "/ 100", changez la derniere
    ligne de la fonction.

    Parametre :
        rows : liste de dictionnaires renvoyee par load_rows().
    Retour :
        un nombre (float) en %, ou None si la collecte totale vaut 0.
    Exemple :
        taux_recyclage(rows)  ->  par exemple 12.5 (pour 12,5 %)
    """
    collecte = sum(row["collecte_kg"] for row in rows)
    if collecte == 0:
        return None
    recycle = total_recycle(rows)
    return recycle / collecte * 100


def collecte_maximum(rows):
    """Retourne la plus grande valeur de collecte_kg sur une seule ligne.

    [TO CONFIRM] "Collecte maximum" est interprete comme le maximum de
    collecte_kg parmi les lignes du fichier (un site, une semaine, une
    categorie). Ce n'est pas le maximum d'un total par site ou par semaine.

    Parametre :
        rows : liste de dictionnaires renvoyee par load_rows().
    Retour :
        un entier en kg, ou None si la liste est vide.
    Exemple :
        collecte_maximum(rows)  ->  la plus grande valeur de collecte_kg
    """
    if not rows:
        return None
    return max(row["collecte_kg"] for row in rows)


def main():
    """Charge le CSV puis affiche deux appels d'exemple."""
    from rapport_pesees import load_rows  # votre fonction de chargement

    chemin = sys.argv[1] if len(sys.argv) > 1 else "pesees.csv"
    rows = load_rows(chemin)

    # Exemple 1 : sur tout le fichier
    print("Total recycle (kg) :", total_recycle(rows))
    taux = taux_recyclage(rows)
    print("Taux de recyclage :", "n/a" if taux is None else f"{taux:.1f} %")
    print("Collecte maximum (kg) :", collecte_maximum(rows))

    # Exemple 2 : meme calcul sur une liste restreinte aux 3 premieres lignes
    extrait = rows[:3]
    print("--- 3 premieres lignes ---")
    print("Total recycle (kg) :", total_recycle(extrait))
    taux = taux_recyclage(extrait)
    print("Taux de recyclage :", "n/a" if taux is None else f"{taux:.1f} %")
    print("Collecte maximum (kg) :", collecte_maximum(extrait))


if __name__ == "__main__":
    main()
