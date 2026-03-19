"""
src/analysis/mercato.py
------------------------
Ce fichier analyse les OPPORTUNITÉS DE MARCHÉ pour le mercato MPG.

L'idée centrale :
  En MPG, chaque joueur a une cotation (prix en millions).
  L'objectif du mercato est de constituer la meilleure équipe possible
  avec un budget limité de 500M.

  Un joueur peut être :
    - SURCÔTÉ   : son prix est élevé mais ses performances ne justifient pas
                  (ex: une grande star qui est blessée souvent ou en méforme)
    - SOUS-CÔTÉ : son prix est bas mais il performe très bien
                  → C'est une "PÉPITE", une bonne affaire à saisir !

Ce module calcule un ratio "valeur/prix" pour trouver ces pépites
et détecte aussi les joueurs en MONTÉE DE FORME récente.
"""

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
from loguru import logger


# ============================================================
# FONCTION 1 : Calculer le rapport qualité/prix de chaque joueur
# ============================================================

def compute_value_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule un score "valeur pour le prix" pour chaque joueur.

    Formule de base :
        value_score = mpg_score / price

    Exemple concret :
        Joueur A : mpg_score = 8.0, price = 40M → value_score = 0.20
        Joueur B : mpg_score = 7.0, price = 10M → value_score = 0.70
        → Joueur B est une bien meilleure affaire malgré un score légèrement inférieur !

    On catégorise ensuite chaque joueur en "tiers" (niveaux) :
        🔥 Pépite       : top 10% des rapports qualité/prix
        ✅ Bonne affaire : entre le 75e et le 90e percentile
        😐 Correct       : entre le 33e et le 75e percentile
        💀 Surcôté       : en dessous du 33e percentile

    Args:
        df (pd.DataFrame): Dataset avec au minimum les colonnes
                           'mpg_score' (notre score composite) et 'price' (cotation MPG).

    Returns:
        pd.DataFrame: Même dataset enrichi avec les colonnes 'value_score' et 'deal_tier'.
    """
    # Vérification : les colonnes nécessaires sont-elles présentes ?
    if "price" not in df.columns or df["price"].isna().all():
        logger.warning("⚠️ Colonne 'price' absente ou vide. Impossible de calculer value_score.")
        return df  # On retourne le tableau sans modification

    if "mpg_score" not in df.columns:
        logger.warning("⚠️ Colonne 'mpg_score' absente. Lance d'abord compute_ratings().")
        return df

    # On travaille sur une copie pour ne pas modifier le tableau original
    df = df.copy()

    # Calcul du ratio valeur/prix
    # .replace(0, np.nan) évite la division par zéro pour les joueurs à prix=0
    df["value_score"] = (
        df["mpg_score"] / df["price"].replace(0, np.nan)
    ).round(3)

    # Calcul des percentiles pour définir les seuils des catégories
    # percentile = valeur en dessous de laquelle se trouve X% des données
    q33 = df["value_score"].quantile(0.33)  # 33% des joueurs ont un score inférieur
    q75 = df["value_score"].quantile(0.75)  # 75% des joueurs ont un score inférieur
    q90 = df["value_score"].quantile(0.90)  # 90% des joueurs ont un score inférieur

    # pd.cut() divise une colonne en intervalles et assigne une étiquette à chacun
    df["deal_tier"] = pd.cut(
        df["value_score"],
        bins=[-np.inf, q33, q75, q90, np.inf],  # Les 4 intervalles (-∞ à q33, q33 à q75, etc.)
        labels=["💀 Surcôté", "😐 Correct", "✅ Bonne affaire", "🔥 Pépite"],  # Étiquettes
    )

    # On trie par value_score décroissant (les meilleures affaires en premier)
    return df.sort_values("value_score", ascending=False)


# ============================================================
# FONCTION 2 : Trouver les meilleures affaires selon des critères
# ============================================================

def find_undervalued(
    df: pd.DataFrame,
    position: str = None,
    max_price: float = None,
    min_score: float = None,
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Retourne les joueurs sous-cotés selon les critères souhaités.

    Cas d'usage typique :
        "Je cherche un milieu qui coûte moins de 20M mais qui performe bien"
        → find_undervalued(df, position="MF", max_price=20)

    Args:
        df (pd.DataFrame): Dataset avec value_score calculé.
                           (utilise d'abord compute_value_score())
        position (str | None): Filtrer par poste : 'GK', 'DF', 'MF', 'FW'.
                               None = toutes les positions.
        max_price (float | None): Prix maximum accepté (en millions MPG).
                                  None = pas de limite de prix.
        min_score (float | None): Score composite minimum.
                                  None = pas de limite de score.
        top_n (int): Nombre de joueurs à retourner.

    Returns:
        pd.DataFrame: Les meilleures affaires qui correspondent aux critères.
    """
    # On vérifie que value_score existe
    if "value_score" not in df.columns:
        logger.warning("⚠️ value_score manquant. Lance d'abord compute_value_score().")
        return pd.DataFrame()

    result = df.copy()

    # Application des filtres successifs
    if position:
        # str.startswith() permet de filtrer "DF" même si le poste vaut "DF latéral"
        result = result[result["position"].str.startswith(position, na=False)]

    if max_price is not None:
        result = result[result["price"] <= max_price]

    if min_score is not None:
        result = result[result["mpg_score"] >= min_score]

    # Sélection des colonnes à afficher (seulement celles qui existent)
    desired_cols = [
        "player_name", "team", "position",
        "avg_rating",       # Note MPG officielle
        "recent_form_avg",  # Forme des 5 dernières journées
        "mpg_score",        # Notre score composite
        "price",            # Cotation MPG
        "value_score",      # Rapport qualité/prix
        "deal_tier",        # Catégorie (pépite, surcôté, etc.)
    ]
    cols = [c for c in desired_cols if c in result.columns]

    logger.info(
        f"🔍 Recherche : position={position}, max_price={max_price}M, "
        f"min_score={min_score} → {len(result)} joueurs trouvés"
    )

    return result[cols].head(top_n)


# ============================================================
# FONCTION 3 : Détecter les joueurs en montée de forme
# ============================================================

def detect_trending_players(
    df_with_history: pd.DataFrame,
    window: int = 5,
    min_games: int = 3,
) -> pd.DataFrame:
    """
    Identifie les joueurs dont la forme récente dépasse leur moyenne de saison.

    Logique :
      - On compare la moyenne des `window` dernières journées à la moyenne de saison
      - Un écart positif = le joueur est EN FORME (à acheter au mercato)
      - Un écart négatif = le joueur est EN MÉFORME (à éviter ou vendre)

    Cette fonction nécessite que le dataset contienne l'historique des notes
    (colonnes 'avg_rating' pour la saison et 'recent_form_avg' pour la forme récente).

    Args:
        df_with_history (pd.DataFrame): Dataset avec avg_rating ET recent_form_avg.
        window (int): Nombre de journées pour calculer la "forme récente". Défaut = 5.
        min_games (int): Minimum de matchs récents pour être inclus. Défaut = 3.

    Returns:
        pd.DataFrame: Joueurs triés par écart de forme décroissant.
    """
    # Vérification des colonnes nécessaires
    required = ["avg_rating", "recent_form_avg"]
    missing = [col for col in required if col not in df_with_history.columns]

    if missing:
        logger.warning(f"⚠️ Colonnes manquantes pour l'analyse de forme : {missing}")
        return pd.DataFrame()

    df = df_with_history.copy()

    # Calcul de l'écart entre forme récente et moyenne de saison
    # Un écart positif = le joueur fait mieux que sa normale = en forme
    df["form_delta"] = (df["recent_form_avg"] - df["avg_rating"]).round(2)

    # Classification de la tendance
    df["trend"] = df["form_delta"].apply(
        lambda delta:
            "🔥 En forme"   if delta > 0.5 else   # +0.5 de moyenne = nette amélioration
            "📈 Progresse"  if delta > 0 else      # Légère amélioration
            "📉 En baisse"  if delta > -0.5 else   # Légère baisse
            "❄️ En méforme"                        # -0.5 ou plus = nette dégradation
    )

    # Colonnes à afficher
    desired_cols = [
        "player_name", "team", "position",
        "avg_rating",       # Moyenne de saison
        "recent_form_avg",  # Moyenne récente
        "form_delta",       # Écart (positif = en forme)
        "trend",            # Catégorie de tendance
        "price",
        "mpg_score",
    ]
    cols = [c for c in desired_cols if c in df.columns]

    return (
        df[cols]
        .dropna(subset=["form_delta"])       # On enlève les joueurs sans données récentes
        .sort_values("form_delta", ascending=False)  # Meilleure forme en premier
    )


# ============================================================
# FONCTION 4 : Rapport mercato complet
# ============================================================

def generate_mercato_report(df: pd.DataFrame) -> dict:
    """
    Génère un rapport complet avec les recommandations mercato.

    Cette fonction est un point d'entrée pratique qui appelle les autres
    fonctions de ce module et retourne un dictionnaire avec tout :
      - Les pépites par poste
      - Les joueurs en forme
      - Les joueurs à éviter (surcotés + en méforme)

    Args:
        df (pd.DataFrame): Dataset master avec tous les scores calculés.

    Returns:
        dict: Dictionnaire avec clés :
              'pepites_gk', 'pepites_df', 'pepites_mf', 'pepites_fw',
              'en_forme', 'a_eviter'
    """
    logger.info("📊 Génération du rapport mercato complet...")

    # Étape 1 : Calculer les scores valeur/prix
    df_valued = compute_value_score(df)

    # Étape 2 : Trouver les pépites par poste
    # On cherche des joueurs pas trop chers (budget raisonnable) et bien notés
    report = {
        "pepites_gk": find_undervalued(df_valued, position="GK", max_price=20, top_n=5),
        "pepites_df": find_undervalued(df_valued, position="DF", max_price=25, top_n=8),
        "pepites_mf": find_undervalued(df_valued, position="MF", max_price=25, top_n=8),
        "pepites_fw": find_undervalued(df_valued, position="FW", max_price=30, top_n=8),
    }

    # Étape 3 : Joueurs en forme (si les données historiques sont disponibles)
    if "recent_form_avg" in df.columns:
        report["en_forme"] = detect_trending_players(df_valued).head(10)

    # Étape 4 : Joueurs à éviter (surcotés = mauvais rapport qualité/prix)
    if "deal_tier" in df_valued.columns:
        report["a_eviter"] = (
            df_valued[df_valued["deal_tier"] == "💀 Surcôté"]
            [["player_name", "team", "position", "mpg_score", "price", "value_score"]]
            .head(10)
        )

    logger.success("✅ Rapport mercato généré.")
    return report


# ============================================================
# POINT D'ENTRÉE — test avec données fictives
# ============================================================

if __name__ == "__main__":
    # Dataset fictif pour tester le module
    sample = pd.DataFrame({
        "player_name":    ["Lacazette", "Thauvin", "Guendouzi", "Nkunku", "Saliba", "Areola"],
        "team":           ["Lyon", "OM", "OM", "Leipzig", "Arsenal", "WHU"],
        "position":       ["FW", "FW", "MF", "MF", "DF", "GK"],
        "avg_rating":     [6.8, 6.5, 6.9, 7.5, 7.1, 6.7],  # Notes MPG officielles
        "recent_form_avg":[7.2, 6.0, 7.5, 8.0, 7.0, 6.5],  # Forme récente
        "mpg_score":      [8.2, 7.5, 7.8, 8.9, 7.1, 6.8],
        "price":          [25, 30, 15, 50, 20, 10],
    })

    print("=" * 55)
    print("  ANALYSE VALEUR/PRIX")
    print("=" * 55)
    df_valued = compute_value_score(sample)
    print(df_valued[["player_name", "mpg_score", "price", "value_score", "deal_tier"]].to_string(index=False))

    print("\n" + "=" * 55)
    print("  PÉPITES MILIEUX (budget max 20M)")
    print("=" * 55)
    print(find_undervalued(df_valued, position="MF", max_price=20).to_string(index=False))

    print("\n" + "=" * 55)
    print("  JOUEURS EN FORME")
    print("=" * 55)
    print(detect_trending_players(df_valued).to_string(index=False))
