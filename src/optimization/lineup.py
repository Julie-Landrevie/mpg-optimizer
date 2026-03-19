"""
src/optimization/lineup.py
---------------------------
Ce fichier trouve le MEILLEUR XI possible pour une journée MPG.

Le problème à résoudre :
  Avec un budget de 500M, on veut choisir 11 joueurs (1 GK + des DF + des MF + des FW)
  qui maximisent le score total, tout en respectant les contraintes MPG :
    - Exactement 1 gardien
    - Entre 3 et 5 défenseurs
    - Entre 3 et 5 milieux
    - Entre 1 et 3 attaquants
    - Maximum 3 joueurs du même club
    - Budget limité

  C'est ce qu'on appelle un problème d'OPTIMISATION COMBINATOIRE.
  On utilise la PROGRAMMATION LINÉAIRE (bibliothèque PuLP) pour le résoudre
  efficacement plutôt que de tester toutes les combinaisons possibles
  (il y en aurait des millions !).

Qu'est-ce que la programmation linéaire ?
  C'est une technique mathématique pour trouver la meilleure solution
  à un problème qui a des contraintes. Ici :
    - On veut MAXIMISER le score total (l'objectif)
    - Sous des CONTRAINTES (budget, nombre de joueurs par poste, etc.)
"""

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
from loguru import logger

# On essaie d'importer PuLP (la bibliothèque d'optimisation)
# Si elle n'est pas installée, on l'indique proprement au lieu de crasher
try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False
    logger.warning("⚠️ PuLP non installé. Installe-le avec : pip install pulp")


# ============================================================
# CONFIGURATION
# ============================================================

# Les formations valides dans MPG
# Chaque tuple = (nombre de défenseurs, nombre de milieux, nombre d'attaquants)
# Total = toujours 10 joueurs de champ + 1 gardien = 11
VALID_FORMATIONS = [
    (3, 4, 3),  # 3-4-3 : très offensif
    (3, 5, 2),  # 3-5-2 : milieu chargé
    (4, 3, 3),  # 4-3-3 : classique offensif
    (4, 4, 2),  # 4-4-2 : classique équilibré
    (4, 5, 1),  # 4-5-1 : défensif avec 1 pivot
    (5, 3, 2),  # 5-3-2 : très défensif
    (5, 4, 1),  # 5-4-1 : le plus défensif
]


# ============================================================
# FONCTION PRINCIPALE : Trouver le meilleur XI
# ============================================================

def optimize_xi(
    df: pd.DataFrame,
    budget: float = 500.0,
    formation: tuple = None,
    locked_players: list = None,
    excluded_players: list = None,
) -> dict:
    """
    Trouve le meilleur XI selon le budget et les contraintes données.

    Si aucune formation n'est précisée, la fonction teste TOUTES les
    formations valides MPG et garde celle qui donne le meilleur score.

    Exemples d'utilisation :
        # XI optimal avec budget standard (toutes formations)
        result = optimize_xi(df, budget=500)

        # XI en 4-4-2 avec budget réduit
        result = optimize_xi(df, budget=300, formation=(4, 4, 2))

        # Avec Lacazette imposé et Mbappe exclu
        result = optimize_xi(df, locked_players=["Lacazette"], excluded_players=["Mbappé"])

    Args:
        df (pd.DataFrame): Dataset avec au minimum : player_name, team, position,
                           mpg_score, price.
        budget (float): Budget total disponible en millions MPG. Défaut = 500.
        formation (tuple | None): Ex: (4, 4, 2) pour 4 déf, 4 mil, 2 att.
                                  None = teste toutes les formations.
        locked_players (list): Liste de noms de joueurs à inclure obligatoirement.
        excluded_players (list): Liste de noms de joueurs à exclure.

    Returns:
        dict: Résultat avec les clés :
              'formation' : la formation utilisée (ex: "4-4-2")
              'players'   : DataFrame des 11 joueurs sélectionnés
              'total_score' : score total du XI
              'total_price' : budget utilisé
    """
    # Vérification que PuLP est disponible
    if not HAS_PULP:
        raise ImportError("PuLP requis pour l'optimisation. Installe-le : pip install pulp")

    # Valeurs par défaut pour les listes
    locked_players   = locked_players or []
    excluded_players = excluded_players or []

    # On retire les joueurs exclus du dataset
    df_filtered = df[~df["player_name"].isin(excluded_players)].copy()

    # Si une formation est spécifiée, on l'utilise seule ; sinon on teste tout
    formations_to_try = [formation] if formation else VALID_FORMATIONS

    # Variables pour stocker le meilleur résultat trouvé
    best_result = None
    best_score  = -1  # Score initial impossible pour que n'importe quelle solution soit meilleure

    logger.info(f"🔍 Test de {len(formations_to_try)} formation(s)...")

    # On essaie chaque formation et on garde la meilleure
    for form in formations_to_try:
        result = _solve_lp(df_filtered, budget, form, locked_players)

        # Si une solution a été trouvée et qu'elle est meilleure que la précédente
        if result and result["total_score"] > best_score:
            best_score  = result["total_score"]
            best_result = result

    if best_result is None:
        logger.error(
            "❌ Aucune solution trouvée ! Vérifie que le budget est suffisant "
            "et qu'il y a assez de joueurs par poste dans le dataset."
        )
        return {}

    logger.success(
        f"✅ XI optimal : Formation {best_result['formation']} | "
        f"Score total : {best_result['total_score']:.2f} | "
        f"Budget utilisé : {best_result['total_price']:.1f}M / {budget}M"
    )
    return best_result


# ============================================================
# FONCTION INTERNE : Résoudre le problème d'optimisation pour une formation
# ============================================================

def _solve_lp(
    df: pd.DataFrame,
    budget: float,
    formation: tuple,
    locked_players: list,
) -> dict | None:
    """
    Résout le problème d'optimisation linéaire pour une formation donnée.

    Cette fonction est "privée" (préfixe _) car elle est un détail d'implémentation.
    Elle est appelée uniquement par optimize_xi().

    Comment fonctionne la programmation linéaire ici ?
      On crée une variable binaire x[i] pour chaque joueur i :
        x[i] = 1 signifie "le joueur i est dans le XI"
        x[i] = 0 signifie "le joueur i n'est pas dans le XI"

      L'objectif est de MAXIMISER : sum(x[i] * mpg_score[i]) pour tous les joueurs i

      Sous les contraintes :
        sum(x[i]) = 11                          → exactement 11 joueurs
        sum(x[i] for GK) = 1                    → exactement 1 gardien
        sum(x[i] * price[i]) <= budget           → budget respecté
        sum(x[i] for même club) <= 3             → max 3 joueurs du même club
        x[locked_player] = 1                    → joueurs imposés

    Args:
        df (pd.DataFrame): Dataset des joueurs disponibles.
        budget (float): Budget maximum.
        formation (tuple): (n_def, n_mid, n_att).
        locked_players (list): Joueurs à inclure obligatoirement.

    Returns:
        dict: Résultat si une solution est trouvée, None sinon.
    """
    n_def, n_mid, n_att = formation  # On décompose le tuple

    # Nombre de joueurs requis par position
    position_counts = {
        "GK": 1,
        "DF": n_def,
        "MF": n_mid,
        "FW": n_att,
    }

    # Création du problème d'optimisation PuLP
    # LpMaximize = on veut maximiser l'objectif (le score)
    prob = pulp.LpProblem(f"MPG_XI_{n_def}{n_mid}{n_att}", pulp.LpMaximize)

    # Création d'une variable binaire (0 ou 1) pour chaque joueur
    # Clé = index du joueur dans le DataFrame, Valeur = variable binaire
    players = df.index.tolist()
    x = pulp.LpVariable.dicts("select", players, cat="Binary")

    # --- OBJECTIF : maximiser le score total du XI ---
    # sum(x[i] * score[i]) pour tous les joueurs
    prob += pulp.lpSum(x[i] * df.loc[i, "mpg_score"] for i in players)

    # --- CONTRAINTES ---

    # Contrainte 1 : Budget total
    if "price" in df.columns:
        prob += (
            pulp.lpSum(x[i] * df.loc[i, "price"] for i in players) <= budget,
            "Budget"  # Nom de la contrainte (optionnel, pour le débogage)
        )

    # Contrainte 2 : Nombre total de joueurs = 11
    prob += pulp.lpSum(x[i] for i in players) == sum(position_counts.values()), "TotalPlayers"

    # Contrainte 3 : Nombre exact de joueurs par position
    for pos, count in position_counts.items():
        # On filtre les joueurs de ce poste
        pos_players = df[df["position"].str.startswith(pos, na=False)].index
        prob += (
            pulp.lpSum(x[i] for i in pos_players) == count,
            f"Position_{pos}"
        )

    # Contrainte 4 : Maximum 3 joueurs du même club
    for team in df["team"].unique():
        team_players = df[df["team"] == team].index
        prob += (
            pulp.lpSum(x[i] for i in team_players) <= 3,
            f"MaxPerTeam_{team}"
        )

    # Contrainte 5 : Joueurs imposés (locked)
    for player_name in locked_players:
        idx = df[df["player_name"] == player_name].index
        if len(idx) > 0:
            prob += x[idx[0]] == 1, f"Locked_{player_name}"
        else:
            logger.warning(f"⚠️ Joueur imposé '{player_name}' non trouvé dans le dataset.")

    # --- RÉSOLUTION ---
    # msg=0 = mode silencieux (pas de logs de PuLP dans le terminal)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Vérification du statut de la solution
    if pulp.LpStatus[prob.status] != "Optimal":
        # Pas de solution optimale trouvée pour cette formation
        return None

    # Extraction des joueurs sélectionnés (ceux avec x[i] = 1)
    selected_indices = [i for i in players if x[i].value() == 1]
    selected_df = df.loc[selected_indices].copy()

    return {
        "formation":   f"{n_def}-{n_mid}-{n_att}",
        "players":     selected_df.sort_values("position"),  # Trié par poste
        "total_score": round(selected_df["mpg_score"].sum(), 2),
        "total_price": round(selected_df["price"].sum(), 1) if "price" in df.columns else None,
    }


# ============================================================
# FONCTION D'AFFICHAGE : Afficher le XI joliment dans le terminal
# ============================================================

def print_xi(result: dict) -> None:
    """
    Affiche le XI sélectionné de façon lisible dans le terminal.

    Exemple de sortie :
        ==================================================
          🏆 XI OPTIMAL — Formation 4-3-3
          Score total : 73.50 | Budget : 412.0M / 500M
        ==================================================
          [GK] Areola            West Ham        ★ 7.2
          [DF] Saliba            Arsenal         ★ 8.1
          ...

    Args:
        result (dict): Le dictionnaire retourné par optimize_xi().
    """
    if not result:
        print("❌ Pas de XI à afficher.")
        return

    # En-tête
    print(f"\n{'='*55}")
    print(f"  🏆 XI OPTIMAL — Formation {result['formation']}")
    print(f"  Score total : {result['total_score']:.2f}", end="")
    if result.get("total_price"):
        print(f" | Budget utilisé : {result['total_price']:.1f}M")
    else:
        print()  # Retour à la ligne si pas de prix
    print(f"{'='*55}")

    df = result["players"]

    # Affichage par ligne de position
    for pos in ["GK", "DF", "MF", "FW"]:
        # On filtre les joueurs de ce poste
        pos_players = df[df["position"].str.startswith(pos, na=False)]

        for _, row in pos_players.iterrows():
            # Formatage du prix si disponible
            price_str = f"  {row['price']:.0f}M" if "price" in df.columns else ""

            # Note MPG officielle si disponible
            rating_str = f"  note: {row['avg_rating']:.1f}" if "avg_rating" in df.columns else ""

            print(
                f"  [{pos}] {row['player_name']:<20} "
                f"{row['team']:<15} "
                f"★ {row['mpg_score']:.1f}"
                f"{price_str}{rating_str}"
            )

    print(f"{'='*55}\n")


# ============================================================
# POINT D'ENTRÉE — test avec données fictives
# ============================================================

if __name__ == "__main__":
    # Génération d'un dataset fictif pour tester l'optimiseur
    import random
    random.seed(42)  # Pour avoir des résultats reproductibles

    positions = ["GK"] * 3 + ["DF"] * 8 + ["MF"] * 10 + ["FW"] * 9

    sample = pd.DataFrame({
        "player_name": [f"Joueur_{i:02d}" for i in range(30)],
        "team":        [f"Club_{i % 8}" for i in range(30)],   # 8 clubs différents
        "position":    positions,
        "mpg_score":   np.random.uniform(4.0, 9.0, 30).round(2),
        "avg_rating":  np.random.uniform(5.0, 8.0, 30).round(1),  # Note MPG officielle
        "price":       np.random.uniform(5.0, 40.0, 30).round(1),
        "minutes":     [1800] * 30,
    })

    print("Test de l'optimiseur de XI...")
    print(f"Dataset : {len(sample)} joueurs disponibles\n")

    # Test avec formation fixe
    result = optimize_xi(sample, budget=500, formation=(4, 4, 2))
    print_xi(result)

    # Test sans formation spécifiée (cherche la meilleure)
    print("\nRecherche de la meilleure formation automatiquement...")
    result_auto = optimize_xi(sample, budget=500)
    print_xi(result_auto)
