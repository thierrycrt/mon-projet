"""Courtieu - Bloc2 : pipeline de nettoyage.

Étapes, dans l'ordre : typage, doublons, manquants, libellés, périmètre.
Les points non décidés sont marqués [TO CONFIRM] et ne modifient pas les données.
"""
import pandas as pd

COLONNES_NUMERIQUES = [
    "collecte_kg",
    "revendu_kg",
    "recycle_kg",
    "dechetterie_kg",
    "attente_kg",
    "semaine",
]
COLONNES_POIDS = [c for c in COLONNES_NUMERIQUES if c.endswith("_kg")]
COLONNE_TOTAL = "collecte_kg"
COLONNES_COMPOSANTES = [c for c in COLONNES_POIDS if c != COLONNE_TOTAL]
CLE_METIER = ["site", "semaine", "categorie"]
CLE_IMPUTATION = ["site", "categorie"]
COLONNE_SITE = "site"
SITE_CONSERVE = "Montreuil"
COLONNE_LIBELLE = "categorie"  # [TO CONFIRM]
DICT_LIBELLES = {"Textiles": "Textile"}
LIBELLES_CONSERVES = ["Papier", "Plastique", "Verre", "Métal", "Bois", "Textile"]


def nettoyer(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    # [TO CONFIRM] travail sur une copie pour ne pas modifier le df d'entrée
    df = df.copy()

    # [TO CONFIRM] structure et noms des clés du dictionnaire de comptes
    comptes = {
        "lignes_entree": len(df),
        "supprimees_par_etape": {
            "typage": 0,
            "doublons": None,    # rempli à l'étape 2
            "manquants": 0,      # la stratégie ne supprime aucune ligne
            "libelles": 0,       # l'harmonisation ne supprime aucune ligne
            "perimetre": None,   # rempli à l'étape 5
        },
        "imputees": None,        # rempli à l'étape 3 (par colonne)
        "lignes_sortie": None,
    }

    # --- 1. Typage (décision fournie) ---------------------------------
    # Une colonne absente lève KeyError (pas de saut silencieux).
    for col in COLONNES_NUMERIQUES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- 2. Doublons (décision fournie) -------------------------------
    # Doublons stricts d'abord, puis doublons sur la clé métier (keep="last").
    # « Dernière entrée enregistrée » = dernière dans l'ordre des lignes du df
    # [TO CONFIRM] aucune colonne d'horodatage n'a été désignée.
    n = len(df)
    df = df.drop_duplicates()
    supprimees_strictes = n - len(df)

    n = len(df)
    df = df.drop_duplicates(subset=CLE_METIER, keep="last")
    supprimees_cle = n - len(df)

    comptes["supprimees_par_etape"]["doublons"] = {
        "stricts": supprimees_strictes,
        "cle_metier": supprimees_cle,
        "total": supprimees_strictes + supprimees_cle,
    }

    # --- 3. Manquants (décision fournie) ------------------------------
    # [TO CONFIRM] colonnes concernées : les 5 poids *_kg ; « semaine » n'est
    # pas imputée.
    # Règles, dans cet ordre (toutes calculées sur les valeurs d'origine) :
    #   a) composante manquante, collecte_kg connu, et c'est la seule composante
    #      manquante de la ligne -> composante = collecte_kg - somme des 3 autres ;
    #   b) toute autre composante manquante -> médiane (site, categorie), puis
    #      médiane globale. [TO CONFIRM] si collecte_kg est connu et que
    #      plusieurs composantes manquent, l'assertion 2 peut échouer ;
    #   c) collecte_kg manquant -> somme des 4 composantes (après a et b).
    # Drapeau booléen « <col>_imputed » : True si la valeur a été imputée.
    manquant = {col: df[col].isna() for col in COLONNES_POIDS}
    mediane_groupe = {
        col: df.groupby(CLE_IMPUTATION)[col].transform("median")
        for col in COLONNES_COMPOSANTES
    }
    mediane_globale = {col: df[col].median() for col in COLONNES_COMPOSANTES}

    nb_composantes_manquantes = sum(manquant[c].astype(int) for c in COLONNES_COMPOSANTES)
    deductible = df[COLONNE_TOTAL].notna() & (nb_composantes_manquantes == 1)

    # a) déduction par différence
    for col in COLONNES_COMPOSANTES:
        autres = [c for c in COLONNES_COMPOSANTES if c != col]
        masque = manquant[col] & deductible
        df.loc[masque, col] = df.loc[masque, COLONNE_TOTAL] - df.loc[masque, autres].sum(axis=1)

    # b) médiane pour les composantes restantes
    for col in COLONNES_COMPOSANTES:
        df[col] = df[col].fillna(mediane_groupe[col]).fillna(mediane_globale[col])

    # c) collecte_kg manquant = somme des composantes (NaN si l'une reste NaN)
    masque = manquant[COLONNE_TOTAL]
    df.loc[masque, COLONNE_TOTAL] = df.loc[masque, COLONNES_COMPOSANTES].sum(axis=1, skipna=False)

    imputees = {}
    for col in COLONNES_POIDS:
        df[f"{col}_imputed"] = manquant[col] & df[col].notna()
        imputees[col] = int(df[f"{col}_imputed"].sum())
    comptes["imputees"] = imputees

    # --- 4. Libellés (décision fournie) -------------------------------
    # Harmonisation par dictionnaire ; les libellés conservés restent tels quels.
    # [TO CONFIRM] colonne concernée : je prends « categorie » (non précisé).
    # Aucune normalisation (strip, casse, accents) : correspondance exacte.
    avant = df[COLONNE_LIBELLE]
    harmonises = int(avant.isin(DICT_LIBELLES.keys()).sum())
    df[COLONNE_LIBELLE] = avant.replace(DICT_LIBELLES)

    # Inconnus = valeurs non vides ni dans le dictionnaire, ni conservées.
    # [TO CONFIRM] remontée : ici dans le dictionnaire de comptes ; les
    # valeurs inconnues sont laissées en l'état (pas d'erreur, pas de recodage).
    connus = set(DICT_LIBELLES) | set(LIBELLES_CONSERVES)
    inconnus = df.loc[df[COLONNE_LIBELLE].notna() & ~df[COLONNE_LIBELLE].isin(connus),
                      COLONNE_LIBELLE]
    comptes["libelles"] = {
        "harmonises": harmonises,
        "inconnus": inconnus.value_counts().to_dict(),
    }

    # --- 5. Périmètre (décision fournie) ------------------------------
    # On ne conserve que le site Montreuil (correspondance exacte ;
    # une ligne dont le site est vide est donc exclue).
    n = len(df)
    df = df[df[COLONNE_SITE] == SITE_CONSERVE]
    comptes["supprimees_par_etape"]["perimetre"] = n - len(df)

    comptes["lignes_sortie"] = len(df)

    # --- Assertions ---------------------------------------------------
    cols_poids = COLONNES_POIDS

    assert (df[cols_poids] >= 0).all().all()
    assert (df["collecte_kg"] == df["revendu_kg"] + df["recycle_kg"] + df["dechetterie_kg"] + df["attente_kg"]).all()
    assert not df.duplicated(subset=["site", "semaine", "categorie"]).any()
    assert df[cols_poids].isna().sum().sum() == 0

    return df, comptes
