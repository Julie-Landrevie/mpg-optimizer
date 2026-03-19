"""
src/analysis/player_rating.py
------------------------------
Ce fichier calcule un SCORE COMPOSITE pour chaque joueur.

L'idée centrale :
  MPG donne déjà une note officielle (avg_rating). Mais cette note seule
  ne suffit pas pour de bonnes décisions car elle ne tient pas compte du PRIX.
  Un joueur à 6.5 de moyenne qui coûte 5M est une bien meilleure affaire
  qu'un joueur à 7.0 de moyenne qui coûte 40M !

Ce module produit deux choses :
  1. Des métriques "per90" (rapportées à 90 minutes) — pour comparer équitablement
     des joueurs avec des temps de jeu différents.
  2. Un score composite qui COMBINE la note MPG officielle + les stats FBref
     pour avoir une image plus complète et plus prédictive.
"""

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd   # Manipulation de tableaux de données
import numpy as np    # Calculs mathématiques (moyennes, normalisation...)
from loguru import logger  # Messages dans le terminal


# ============================================================
# CONFIGURATION : les poids par position pour le score composite
# ============================================================

# Dictionnaire qui définit l'importance (poids) de chaque statistique
# selon le poste du joueur.
#
# Logique des poids :
#   - Les poids positifs : plus la stat est élevée, mieux c'est
#   - Les poids négatifs : plus la stat est élevée, moins c'est bien
#   - La somme des valeurs absolues des poids = 1.0 (100%)
#   - Plus le poids est élevé, plus la stat compte dans le score final
#
# Note importante : on intègre "avg_rating" (la note MPG officielle) comme
# l'une des statistiques, avec un poids important. C'est le principal
# enrichissement par rapport à la version précédente du code.

POSITION_WEIGHTS = {
    # ---- GARDIENS ----
    # Pour un gardien, les clean sheets et les arrêts comptent le plus
    "GK": {
        "avg_rating":            0.45,  # La note MPG officielle est très fiable pour les GK
        "clean_sheets_per90":    0.30,  # Matchs sans encaisser de but / 90 min
        "saves_per90":           0.15,  # Arrêts / 90 min
        "goals_against_per90":  -0.10,  # Buts encaissés / 90 min (négatif = c'est mauvais)
    },

    # ---- DÉFENSEURS ----
    "DF": {
        "avg_rating":             0.40,  # La note MPG reste la base
        "clean_sheets_per90":     0.25,  # Très important pour un défenseur
        "xAG_per90":              0.15,  # Contribution offensive (passes décisives attendues)
        "progressive_carries_per90": 0.10,  # Portées de balle vers l'avant
        "tackles_won_per90":      0.10,  # Tacles réussis
    },

    # ---- MILIEUX ----
    "MF": {
        "avg_rating":                 0.35,
        "xAG_per90":                  0.25,  # Expected assists = contribution créative
        "npxG_per90":                 0.15,  # xG sans pénaltys (contributions offensives)
        "progressive_passes_per90":   0.15,  # Passes qui avancent vers le but adverse
        "key_passes_per90":           0.10,  # Passes qui mènent à un tir
    },

    # ---- ATTAQUANTS ----
    "FW": {
        "avg_rating":            0.30,
        "npxG_per90":            0.30,  # xG hors pénaltys — prédit les futurs buts
        "goals_per90":           0.20,  # Vrais buts marqués
        "xAG_per90":             0.15,  # Passes décisives attendues
        "shots_on_target_per90": 0.05,  # Tirs cadrés — montre l'activité offensive
    },
}


# ============================================================
# FONCTIONS DE CALCUL
# ============================================================

def normalize_series(s: pd.Series) -> pd.Series:
    """
    Normalise une série de valeurs entre 0 et 10.

    Pourquoi normaliser ?
      Les statistiques ont des échelles très différentes :
        - goals_per90 varie entre 0 et 1
        - progressive_passes_per90 peut aller jusqu'à 10+
      Sans normalisation, les grandes valeurs écraseraient les petites.
      Après normalisation, toutes les stats ont la même importance de base.

    Formule : (valeur - minimum) / (maximum - minimum) × 10

    Exemples :
        [1, 2, 3, 4, 5]  →  [0, 2.5, 5, 7.5, 10]
        [3, 3, 3]        →  [5, 5, 5]  (valeur constante → milieu = 5)

    Args:
        s (pd.Series): La série de valeurs à normaliser.

    Returns:
        pd.Series: Valeurs normalisées entre 0 et 10.
    """
    # Cas particulier : si toutes les valeurs sont identiques (pas de variation),
    # on retourne 5.0 pour tous (valeur neutre au milieu de l'échelle 0-10)
    if s.max() == s.min():
        return pd.Series(5.0, index=s.index)

    # Formule min-max standard
    return (s - s.min()) / (s.max() - s.min()) * 10


def compute_per90(df: pd.DataFrame, col: str, minutes_col: str = "minutes") -> pd.Series:
    """
    Calcule une statistique rapportée à 90 minutes de jeu.

    Pourquoi "per 90" ?
      Un joueur qui joue 10 matchs complets (900 min) et marque 3 buts
      a 0.3 buts/90. Un remplaçant qui joue 200 min et marque 1 but
      a 0.45 buts/90. La statistique per90 est plus juste pour comparer.

    Formule : stat_totale / (minutes_jouées / 90)

    Args:
        df (pd.DataFrame): Le tableau de données.
        col (str): Nom de la colonne à diviser (ex: "goals").
        minutes_col (str): Nom de la colonne des minutes jouées.

    Returns:
        pd.Series: La statistique rapportée à 90 minutes.
    """
    # .replace(0, np.nan) évite une division par zéro pour les joueurs sans minutes
    # np.nan = "Not a Number" = valeur manquante en Python
    return df[col] / (df[minutes_col] / 90).replace(0, np.nan)


def compute_ratings(df: pd.DataFrame, min_minutes: int = 450) -> pd.DataFrame:
    """
    Calcule le score MPG composite pour chaque joueur.

    Ce score combine :
      - La note MPG officielle (avg_rating) — pour les performances globales
      - Les statistiques FBref per90 — pour les détails tactiques
      - Pondération différente selon le poste du joueur

    Args:
        df (pd.DataFrame): Dataset master avec stats MPG + FBref.
                           Doit contenir au minimum : player_name (ou player),
                           position, minutes, avg_rating.
        min_minutes (int): Minutes minimum pour être inclus dans l'analyse.
                           Par défaut 450 = 5 matchs complets.
                           Un joueur avec moins de 450 min a trop peu de données.

    Returns:
        pd.DataFrame: Même tableau enrichi avec les colonnes :
                      - [stat]_per90 : chaque stat rapportée à 90 min
                      - mpg_score : le score composite sur 10
                      - rank : classement au sein de sa position
    """
    logger.info(f"Calcul des scores pour {len(df)} joueurs (seuil : {min_minutes} min)...")

    # --- Étape 1 : Filtrer les joueurs sans assez de minutes ---
    # Un joueur qui a joué 50 minutes a des statistiques trop peu fiables
    df = df[df["minutes"] >= min_minutes].copy()
    # .copy() crée une copie indépendante pour éviter les avertissements pandas
    logger.info(f"Joueurs retenus après filtre minutes : {len(df)}")

    # --- Étape 2 : Calculer les versions "per90" des statistiques ---
    # On liste toutes les stats pour lesquelles on veut la version per90
    per90_stats = [
        "goals",                # Buts
        "assists",              # Passes décisives
        "npxG",                 # Expected goals hors pénaltys
        "xAG",                  # Expected assists
        "shots_on_target",      # Tirs cadrés
        "key_passes",           # Passes clés (qui mènent à un tir)
        "progressive_passes",   # Passes progressives (vers l'avant)
        "progressive_carries",  # Portées de balle progressives
        "clean_sheets",         # Matchs sans encaisser (gardiens/défenseurs)
        "saves",                # Arrêts (gardiens)
        "goals_against",        # Buts encaissés (gardiens)
        "tackles_won",          # Tacles réussis
    ]

    # Pour chaque stat, on crée une nouvelle colonne [stat]_per90 si la colonne source existe
    for stat in per90_stats:
        if stat in df.columns:
            df[f"{stat}_per90"] = compute_per90(df, stat)

    # --- Étape 3 : Calculer le score composite par position ---
    # On initialise la colonne du score à 0 pour tous les joueurs
    df["mpg_score"] = 0.0

    # On boucle sur chaque groupe de position (GK, DF, MF, FW)
    for position_prefix, weights in POSITION_WEIGHTS.items():

        # On sélectionne les joueurs de cette position
        # str.startswith() permet de gérer "DF", "DF latéral", etc.
        mask = df["position"].str.startswith(position_prefix, na=False)
        pos_df = df[mask]  # Sous-tableau des joueurs de cette position

        if pos_df.empty:
            continue  # On passe si aucun joueur à cette position

        # On calcule le score comme une somme pondérée des statistiques normalisées
        score = pd.Series(0.0, index=pos_df.index)  # Score initial = 0

        for stat, weight in weights.items():
            if stat not in df.columns:
                # La statistique n'est pas disponible (FBref non récupéré, etc.)
                logger.debug(f"Stat manquante : {stat} — ignorée pour {position_prefix}")
                continue

            # On remplace les valeurs manquantes (NaN) par 0 avant la normalisation
            stat_values = pos_df[stat].fillna(0)

            # On normalise entre 0 et 10
            normalized = normalize_series(stat_values)

            # On ajoute la contribution de cette stat au score
            # abs(weight) = valeur absolue du poids (toujours positif)
            # np.sign(weight) = +1 ou -1 selon le signe du poids
            # Exemple avec goals_against (poids négatif) :
            #   un gardien qui encaisse peu aura une note normalisée HAUTE
            #   mais on la multiplie par -1 pour qu'un haut score = beaucoup de buts encaissés
            #   → contribution négative au score final = c'est cohérent
            score += normalized * abs(weight) * np.sign(weight)

        # On clipe entre 0 et 10 (au cas où la somme dépasse les bornes) et on arrondit
        df.loc[mask, "mpg_score"] = score.clip(0, 10).round(2)

    # --- Étape 4 : Calculer le classement par position ---
    # rank() calcule le rang dans le groupe — le meilleur obtient le rang 1
    df["rank"] = (
        df.groupby("position")["mpg_score"]
        .rank(method="dense", ascending=False)  # "dense" = pas de trous dans la numérotation
        .astype(int)
    )

    # On trie par score décroissant (meilleurs en premier)
    df = df.sort_values("mpg_score", ascending=False)

    logger.success("✅ Scores composites calculés avec succès.")
    return df


def get_top_players(
    df: pd.DataFrame,
    position: str = None,
    top_n: int = 20
) -> pd.DataFrame:
    """
    Retourne les meilleurs joueurs, avec un filtre optionnel par position.

    Args:
        df (pd.DataFrame): DataFrame avec les scores calculés.
        position (str | None): 'GK', 'DF', 'MF' ou 'FW'.
                               None = retourne toutes les positions.
        top_n (int): Combien de joueurs afficher. Défaut = 20.

    Returns:
        pd.DataFrame: Les meilleurs joueurs, triés par score décroissant.
    """
    # On définit les colonnes qu'on veut afficher (si elles existent)
    desired_cols = [
        "player_name", "team", "position", "minutes",
        "avg_rating",       # La note MPG officielle
        "recent_form_avg",  # La forme récente (5 dernières journées)
        "mpg_score",        # Notre score composite
        "price",            # La cotation MPG
        "rank",
    ]
    # On ne garde que les colonnes qui existent réellement dans le DataFrame
    # (certaines peuvent manquer si FBref n'est pas disponible)
    cols = [c for c in desired_cols if c in df.columns]

    result = df[cols].copy()

    # Filtre par position si demandé
    if position:
        result = result[result["position"].str.startswith(position, na=False)]

    # On retourne seulement les top_n premières lignes
    return result.head(top_n)


# ============================================================
# POINT D'ENTRÉE — test avec des données fictives
# ============================================================

if __name__ == "__main__":
    # Création d'un mini-dataset fictif pour tester le code
    # En pratique, on utilisera le dataset réel de build_master_dataset()

    print("Test du module de scoring avec données fictives...\n")

    sample_data = pd.DataFrame({
        "player_name":  ["Lacazette", "Thauvin", "Guendouzi", "Saliba", "Areola"],
        "team":         ["Lyon", "OM", "OM", "Arsenal", "WHU"],
        "position":     ["FW", "FW", "MF", "DF", "GK"],
        "minutes":      [1800, 1600, 1900, 2100, 2000],
        # Note MPG officielle (la vraie note de l'appli)
        "avg_rating":   [6.8, 6.5, 6.9, 7.1, 6.7],
        # Stats FBref
        "goals":        [12, 9, 3, 1, 0],
        "assists":      [4, 6, 8, 2, 0],
        "npxG":         [10.5, 8.2, 2.8, 0.9, 0.1],
        "xAG":          [3.8, 5.1, 7.4, 1.8, 0.3],
        "shots_on_target": [35, 28, 12, 6, 2],
        "key_passes":   [20, 30, 55, 10, 4],
        "price":        [25, 20, 15, 30, 10],
    })

    # On calcule les scores
    rated = compute_ratings(sample_data, min_minutes=0)

    # On affiche les résultats
    print("Résultats du scoring composite :")
    print(rated[["player_name", "position", "avg_rating", "mpg_score", "rank"]].to_string(index=False))

    print("\nTop attaquants :")
    print(get_top_players(rated, position="FW", top_n=5).to_string(index=False))
