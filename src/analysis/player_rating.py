"""
src/analysis/player_rating.py
------------------------------
Ce fichier calcule un SCORE COMPOSITE pour chaque joueur.

Colonnes disponibles dans notre dataset :
  Depuis MPG :
    - avg_rating      : note moyenne de saison (source principale)
    - recent_form_avg : moyenne des 5 dernières notes
    - avg_points      : points MPG moyens
    - games_played    : nombre de matchs joués
    - started_matches : nombre de matchs démarrés
    - total_goals     : buts marqués
    - clean_sheets    : matchs sans encaisser (GK/DF)
    - goals_conceded  : buts encaissés (GK)
    - price           : cotation en millions

  Depuis FBref (disponible pour ~27 joueurs seulement) :
    - Min             : minutes jouées
    - Gls             : buts (total)
    - Ast             : passes décisives
    - Gls.1           : buts per 90 min
    - Ast.1           : passes décisives per 90 min
    - G+A-PK          : buts+passes hors pénaltys per 90

Logique du score :
  La note MPG (avg_rating) est notre pilier central car c'est la source
  la plus complète et la plus fiable (514 joueurs vs 27 pour FBref).
  On l'enrichit avec la forme récente, les stats MPG disponibles,
  et les stats FBref quand elles existent.
"""

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
from loguru import logger


# ============================================================
# CONFIGURATION : poids par position
# ============================================================

# Logique des poids :
#   - avg_rating est la base solide pour tous les postes
#   - recent_form_avg capte les joueurs en montée/baisse de forme
#   - Les stats spécifiques affinent selon le rôle tactique
#   - Tous les poids d'un poste doivent sommer à 1.0

POSITION_WEIGHTS = {

    # ---- GARDIENS ----
    # Un gardien est évalué sur sa solidité défensive.
    # La note MPG est très fiable pour les GK (elle intègre arrêts + clean sheets).
    "GK": {
        "avg_rating":       0.50,  # Note MPG officielle — pilier
        "recent_form_avg":  0.20,  # Forme des 5 dernières journées
        "clean_sheets":     0.20,  # Matchs sans encaisser (bonus MPG important)
        "goals_conceded":  -0.10,  # Buts encaissés (poids négatif = mauvais signe)
    },

    # ---- DÉFENSEURS ----
    # Un défenseur marque des points MPG via les clean sheets et les centres/passes.
    "DF": {
        "avg_rating":       0.50,  # Note MPG officielle — pilier
        "recent_form_avg":  0.25,  # Forme récente
        "clean_sheets":     0.15,  # Clean sheets = bonus MPG défensifs
        "total_goals":      0.10,  # Buts marqués (défenseurs offensifs bien notés)
    },

    # ---- MILIEUX ----
    # Un milieu marque via les passes décisives et les buts.
    # avg_points reflète bien la contribution réelle en MPG.
    "MF": {
        "avg_rating":       0.45,  # Note MPG officielle — pilier
        "recent_form_avg":  0.25,  # Forme récente
        "avg_points":       0.20,  # Points MPG moyens (intègre buts + passes)
        "total_goals":      0.10,  # Buts marqués
    },

    # ---- ATTAQUANTS ----
    # Un attaquant est jugé quasi exclusivement sur les buts et la constance.
    "FW": {
        "avg_rating":       0.40,  # Note MPG officielle — pilier
        "recent_form_avg":  0.25,  # Forme récente (crucial pour les FW chauds)
        "avg_points":       0.20,  # Points MPG moyens
        "total_goals":      0.15,  # Buts marqués sur la saison
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
        - avg_rating varie entre 4 et 8
        - total_goals peut aller jusqu'à 25
        - clean_sheets jusqu'à 20
      Sans normalisation, les grandes valeurs écraseraient les petites.
      Après normalisation, toutes les stats ont la même importance de base,
      et c'est le POIDS qui détermine leur contribution finale.

    Formule : (valeur - minimum) / (maximum - minimum) × 10

    Exemples :
        [4, 5, 6, 7, 8]  →  [0, 2.5, 5, 7.5, 10]
        [3, 3, 3]        →  [5, 5, 5]  (valeur constante → neutre = 5)

    Args:
        s (pd.Series): La série de valeurs à normaliser.

    Returns:
        pd.Series: Valeurs normalisées entre 0 et 10.
    """
    # Cas particulier : si toutes les valeurs sont identiques (pas de variation),
    # on retourne 5.0 pour tous (valeur neutre au milieu de l'échelle 0-10)
    if s.max() == s.min():
        return pd.Series(5.0, index=s.index)

    return (s - s.min()) / (s.max() - s.min()) * 10


def compute_ratings(df: pd.DataFrame, min_minutes: int = 0) -> pd.DataFrame:
    """
    Calcule le score MPG composite pour chaque joueur.

    Ce score combine :
      - La note MPG officielle (avg_rating) — base solide pour tous
      - La forme récente (recent_form_avg) — est-il en train de monter ?
      - Les stats MPG disponibles selon le poste
      - Pondération différente selon le poste du joueur

    Note sur min_minutes :
      Dans notre dataset, on n'a pas de colonne "minutes" directe pour tous
      les joueurs (FBref n'est disponible que pour ~27 joueurs).
      On utilise à la place 'games_played' (matchs joués) pour filtrer
      les joueurs sans assez de données. Par défaut à 0 = tous les joueurs.

    Args:
        df (pd.DataFrame): Dataset master avec colonnes MPG + FBref.
        min_minutes (int): Seuil de matchs joués minimum.
                           0 = tous les joueurs inclus.
                           5 = seulement ceux ayant joué au moins 5 matchs.

    Returns:
        pd.DataFrame: Même tableau enrichi avec les colonnes :
                      - mpg_score : le score composite sur 10
                      - rank      : classement au sein de sa position
    """
    logger.info(f"Calcul des scores pour {len(df)} joueurs (seuil : {min_minutes} matchs)...")

    # --- Filtrage sur les matchs joués ---
    # On utilise 'games_played' (disponible pour tous via MPG)
    # car 'minutes' n'existe que pour les joueurs croisés avec FBref
    if min_minutes > 0 and "games_played" in df.columns:
        df = df[df["games_played"] >= min_minutes].copy()
        logger.info(f"Joueurs retenus après filtre matchs : {len(df)}")
    else:
        df = df.copy()

    # --- Filtrage sur avg_rating disponible ---
    # On ne peut pas scorer un joueur sans note MPG
    df = df[df["avg_rating"].notna()].copy()
    logger.info(f"Joueurs avec note MPG disponible : {len(df)}")

    # --- Calcul du score composite par position ---
    df["mpg_score"] = 0.0

    for position_prefix, weights in POSITION_WEIGHTS.items():

        # On sélectionne les joueurs de cette position
        # str.startswith() gère les variantes comme "MF,DF" dans FBref
        mask = df["position"].str.startswith(position_prefix, na=False)
        pos_df = df[mask]

        if pos_df.empty:
            logger.debug(f"Aucun joueur trouvé pour la position {position_prefix}")
            continue

        logger.debug(f"{position_prefix} : {len(pos_df)} joueurs")

        # Calcul de la somme pondérée des stats normalisées
        score = pd.Series(0.0, index=pos_df.index)

        for stat, weight in weights.items():
            if stat not in df.columns:
                logger.debug(f"Stat manquante : {stat} — ignorée pour {position_prefix}")
                continue

            # On remplace les valeurs manquantes (NaN) par 0 avant normalisation
            stat_values = pos_df[stat].fillna(0)

            # Normalisation entre 0 et 10
            normalized = normalize_series(stat_values)

            # Contribution au score :
            # abs(weight) = valeur absolue du poids (toujours positif pour la normalisation)
            # np.sign(weight) = +1 ou -1 selon le signe du poids
            # Exemple : goals_conceded a un poids négatif (-0.10)
            # Un gardien qui encaisse beaucoup aura une note normalisée HAUTE
            # mais multipliée par -1 → contribution négative au score final ✓
            score += normalized * abs(weight) * np.sign(weight)

        # On clipe entre 0 et 10 et on arrondit à 2 décimales
        df.loc[mask, "mpg_score"] = score.clip(0, 10).round(2)

    # --- Classement par position ---
    # rank() calcule le rang dans le groupe — le meilleur obtient le rang 1
    df["rank"] = (
        df.groupby("position")["mpg_score"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )

    # Tri par score décroissant (meilleurs en premier)
    df = df.sort_values("mpg_score", ascending=False)

    logger.success("✅ Scores composites calculés avec succès.")
    return df


def get_top_players(
    df: pd.DataFrame,
    position: str = None,
    top_n: int = 20,
    min_games: int = 5,
) -> pd.DataFrame:
    """
    Retourne les meilleurs joueurs, avec filtres optionnels.

    Args:
        df (pd.DataFrame): DataFrame avec les scores calculés.
        position (str | None): 'GK', 'DF', 'MF' ou 'FW'. None = toutes positions.
        top_n (int): Nombre de joueurs à retourner. Défaut = 20.
        min_games (int): Matchs minimum joués pour être inclus. Défaut = 5.

    Returns:
        pd.DataFrame: Les meilleurs joueurs, triés par score décroissant.
    """
    # Colonnes à afficher — on prend celles qui existent réellement
    desired_cols = [
        "player_name",      # Nom du joueur
        "first_name",       # Prénom
        "position",         # Poste
        "team",             # Club (identifiant MPG)
        "games_played",     # Matchs joués
        "avg_rating",       # Note MPG officielle de saison
        "recent_form_avg",  # Forme récente (5 dernières journées)
        "mpg_score",        # Notre score composite
        "price",            # Cotation MPG
        "rank",             # Classement dans sa position
        "status",           # Disponible / Blessé / Suspendu
    ]
    cols = [c for c in desired_cols if c in df.columns]

    result = df[cols].copy()

    # Filtre par matchs joués minimum
    if min_games > 0 and "games_played" in result.columns:
        result = result[result["games_played"] >= min_games]

    # Filtre par position si demandé
    if position:
        result = result[result["position"].str.startswith(position, na=False)]

    return result.head(top_n)


def get_value_picks(
    df: pd.DataFrame,
    position: str = None,
    max_price: float = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Trouve les meilleurs joueurs selon leur rapport score/prix.

    C'est une version simplifiée de mercato.py pour avoir rapidement
    les bonnes affaires sans passer par le module complet.

    Args:
        df (pd.DataFrame): DataFrame avec scores calculés.
        position (str | None): Filtrer par poste.
        max_price (float | None): Budget maximum par joueur.
        top_n (int): Nombre de résultats.

    Returns:
        pd.DataFrame: Meilleures affaires triées par rapport score/prix.
    """
    if "mpg_score" not in df.columns or "price" not in df.columns:
        logger.warning("⚠️ mpg_score ou price manquant. Lance d'abord compute_ratings().")
        return pd.DataFrame()

    result = df.copy()

    # Calcul du rapport qualité/prix
    result["value_score"] = (
        result["mpg_score"] / result["price"].replace(0, np.nan)
    ).round(3)

    # Filtres optionnels
    if position:
        result = result[result["position"].str.startswith(position, na=False)]
    if max_price:
        result = result[result["price"] <= max_price]

    # Filtre prix minimum (évite les joueurs à 1M peu utilisés)
    result = result[result["price"] >= 5]

    # Filtre : au moins 10 matchs joués
    if "started_matches" in result.columns:
        result = result[result["started_matches"] >= 8]

    desired_cols = [
        "player_name", "first_name", "position", "team",
        "avg_rating", "recent_form_avg", "mpg_score", "price", "value_score", "status"
    ]
    cols = [c for c in desired_cols if c in result.columns]

    return result[cols].sort_values("value_score", ascending=False).head(top_n)


# ============================================================
# POINT D'ENTRÉE — test avec les vraies données
# ============================================================

if __name__ == "__main__":

    # Import ici pour éviter les imports circulaires
    from src.data.collect import build_master_dataset

    print("=" * 60)
    print("  MPG Optimizer — Scoring des joueurs")
    print("=" * 60)

    # Chargement du dataset master
    df = build_master_dataset()
    print(f"\n📋 Dataset chargé : {len(df)} joueurs, {len(df.columns)} colonnes")

    # Calcul des scores composites
    df_rated = compute_ratings(df, min_minutes=0)

    # ── Top 10 global ──
    print("\n🏆 TOP 10 JOUEURS (toutes positions, min. 5 matchs) :")
    top_all = get_top_players(df_rated, top_n=10, min_games=5)
    print(top_all[["player_name", "position", "avg_rating", "recent_form_avg",
                   "mpg_score", "price"]].to_string(index=False))

    # ── Top 5 par poste ──
    for pos, label in [("GK", "GARDIENS"), ("DF", "DÉFENSEURS"),
                       ("MF", "MILIEUX"), ("FW", "ATTAQUANTS")]:
        print(f"\n⚽ TOP 5 {label} :")
        top_pos = get_top_players(df_rated, position=pos, top_n=5, min_games=5)
        print(top_pos[["player_name", "avg_rating", "recent_form_avg",
                       "mpg_score", "price", "status"]].to_string(index=False))

    # ── Pépites (bon rapport qualité/prix) ──
    print("\n🔥 TOP 5 PÉPITES MILIEUX (budget max 20M) :")
    pepites = get_value_picks(df_rated, position="MF", max_price=20, top_n=5)
    if not pepites.empty:
        print(pepites[["player_name", "avg_rating", "mpg_score",
                       "price", "value_score"]].to_string(index=False))

    print("\n✨ Analyse terminée !")
