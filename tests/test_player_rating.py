"""
tests/test_player_rating.py
----------------------------
Tests unitaires du score MPG (src/analysis/player_rating.py) et de l'optimiseur de XI
(src/optimization/lineup.py).

Réécrits le 24/11/2026 : la première version testait l'ancien module (statistiques FBref
par 90 minutes, fonction compute_per90). Depuis mai 2026, le score repose sur les données
de l'API MPG (note moyenne, forme récente, points, buts, clean sheets) : les tests suivent
désormais ce schéma de données.

Lancer les tests, depuis la racine du projet :
    pytest tests/          # tous les tests
    pytest tests/ -v       # avec le nom de chaque test
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.player_rating import (  # noqa: E402
    POSITION_WEIGHTS,
    compute_ratings,
    get_top_players,
    get_value_picks,
    normalize_series,
)
from src.optimization.lineup import optimize_xi  # noqa: E402


# ============================================================
# FIXTURES : données au format du dataset master (API MPG)
# ============================================================

@pytest.fixture
def mpg_players():
    """Dix joueurs, deux à trois par poste, avec les colonnes MPG réelles."""
    return pd.DataFrame({
        "player_name":     ["Areola", "Lloris", "Saliba", "Marquinhos", "Hakimi", "Guendouzi",
                            "Tolisso", "Lacazette", "Greenwood", "Wahi"],
        "first_name":      ["Alphonse", "Hugo", "William", "Marcos", "Achraf", "Mattéo",
                            "Corentin", "Alexandre", "Mason", "Elye"],
        "team":            ["A", "B", "C", "D", "D", "E", "F", "F", "E", "G"],
        "position":        ["GK", "GK", "DF", "DF", "DF", "MF", "MF", "FW", "FW", "FW"],
        "avg_rating":      [6.7, 5.4, 7.1, 6.6, 6.2, 6.9, 5.8, 6.8, 6.5, 5.2],
        "recent_form_avg": [6.5, 5.0, 7.0, 6.0, 6.8, 7.5, 5.5, 7.2, 6.0, 4.8],
        "avg_points":      [4.0, 2.0, 5.0, 4.5, 4.2, 5.5, 3.0, 6.0, 5.0, 2.0],
        "games_played":    [14, 9, 15, 12, 4, 15, 10, 13, 14, 6],
        "started_matches": [14, 9, 15, 11, 3, 14, 8, 12, 13, 2],
        "total_goals":     [0, 0, 1, 2, 1, 3, 1, 9, 7, 1],
        "clean_sheets":    [6, 1, 6, 5, 2, 0, 0, 0, 0, 0],
        "goals_conceded":  [12, 25, 0, 0, 0, 0, 0, 0, 0, 0],
        "price":           [18, 6, 30, 22, 25, 15, 9, 28, 20, 5],
        "status":          ["Disponible"] * 9 + ["Blessé"],
    })


@pytest.fixture
def mpg_pool():
    """Un vivier de 60 joueurs (assez pour toutes les formations), graine fixe."""
    rng = np.random.default_rng(42)
    positions = ["GK"] * 6 + ["DF"] * 18 + ["MF"] * 20 + ["FW"] * 16
    n = len(positions)
    return pd.DataFrame({
        "player_name":     [f"Joueur_{i:02d}" for i in range(n)],
        "team":            [f"Club_{i % 12}" for i in range(n)],
        "position":        positions,
        "avg_rating":      rng.uniform(4.5, 7.5, n).round(2),
        "recent_form_avg": rng.uniform(4.0, 8.0, n).round(2),
        "avg_points":      rng.uniform(1, 7, n).round(2),
        "games_played":    rng.integers(0, 16, n),
        "started_matches": rng.integers(0, 14, n),
        "total_goals":     rng.integers(0, 10, n),
        "clean_sheets":    rng.integers(0, 8, n),
        "goals_conceded":  rng.integers(0, 25, n),
        "price":           rng.integers(1, 40, n),
    })


# ============================================================
# normalize_series()
# ============================================================

class TestNormalizeSeries:

    def test_output_min_is_zero(self):
        assert normalize_series(pd.Series([2, 5, 9])).min() == pytest.approx(0.0)

    def test_output_max_is_ten(self):
        assert normalize_series(pd.Series([2, 5, 9])).max() == pytest.approx(10.0)

    def test_constant_series_returns_five(self):
        assert (normalize_series(pd.Series([3, 3, 3])) == 5.0).all()

    def test_preserves_order(self):
        s = pd.Series([4.0, 1.0, 7.0, 2.5])
        assert list(normalize_series(s).rank()) == list(s.rank())

    def test_output_length_unchanged(self):
        s = pd.Series(range(17))
        assert len(normalize_series(s)) == len(s)


# ============================================================
# compute_ratings()
# ============================================================

class TestComputeRatings:

    def test_mpg_score_and_rank_created(self, mpg_players):
        rated = compute_ratings(mpg_players)
        assert {"mpg_score", "rank"} <= set(rated.columns)

    def test_scores_within_range(self, mpg_pool):
        rated = compute_ratings(mpg_pool)
        assert rated["mpg_score"].between(0, 10).all()

    def test_sorted_descending(self, mpg_pool):
        scores = compute_ratings(mpg_pool)["mpg_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_min_games_filter(self, mpg_pool):
        """Le paramètre min_minutes filtre sur les matchs joués (games_played)."""
        rated = compute_ratings(mpg_pool, min_minutes=8)
        assert (rated["games_played"] >= 8).all()
        assert len(rated) == (mpg_pool["games_played"] >= 8).sum()

    def test_players_without_rating_are_dropped(self, mpg_players):
        df = mpg_players.copy()
        df.loc[0, "avg_rating"] = np.nan
        rated = compute_ratings(df)
        assert "Areola" not in rated["player_name"].values
        assert len(rated) == len(df) - 1

    def test_input_not_modified(self, mpg_players):
        before = mpg_players.copy()
        compute_ratings(mpg_players, min_minutes=5)
        pd.testing.assert_frame_equal(mpg_players, before)

    def test_best_player_of_position_ranked_first(self, mpg_players):
        """Le meilleur sur toutes les stats d'un poste doit y être classé 1er."""
        rated = compute_ratings(mpg_players)
        gk = rated[rated["position"] == "GK"].set_index("player_name")
        assert gk.loc["Areola", "rank"] == 1
        assert gk.loc["Areola", "mpg_score"] > gk.loc["Lloris", "mpg_score"]

    def test_goals_conceded_counts_against_goalkeeper(self):
        """Poids négatif : à notes égales, le gardien qui encaisse moins est devant."""
        df = pd.DataFrame({
            "player_name": ["Peu", "Beaucoup", "Moyen"], "team": ["A", "B", "C"],
            "position": ["GK"] * 3, "avg_rating": [6.0] * 3, "recent_form_avg": [6.0] * 3,
            "clean_sheets": [3] * 3, "goals_conceded": [5, 30, 15], "price": [10] * 3,
        })
        rated = compute_ratings(df).set_index("player_name")
        assert rated.loc["Peu", "mpg_score"] > rated.loc["Moyen", "mpg_score"] > rated.loc["Beaucoup", "mpg_score"]

    def test_positions_scored_with_their_own_weights(self, mpg_players):
        """Chaque poste a des poids qui somment à 1 en valeur absolue (score max 10)."""
        for pos, weights in POSITION_WEIGHTS.items():
            assert sum(abs(w) for w in weights.values()) == pytest.approx(1.0), pos
        rated = compute_ratings(mpg_players)
        for pos in ["GK", "DF", "MF", "FW"]:
            assert (rated["position"] == pos).any()

    def test_rank_is_within_position(self, mpg_players):
        rated = compute_ratings(mpg_players)
        for pos, grp in rated.groupby("position"):
            assert grp["rank"].min() == 1, pos

    def test_missing_optional_stats(self):
        """Avec seulement la note MPG, le calcul fonctionne quand même."""
        df = pd.DataFrame({"player_name": ["A", "B", "C"], "team": ["X", "Y", "Z"],
                           "position": ["FW", "MF", "DF"], "avg_rating": [7.0, 6.5, 7.2]})
        rated = compute_ratings(df)
        assert len(rated) == 3 and "mpg_score" in rated.columns

    def test_single_player_per_position_is_rank_one(self):
        df = pd.DataFrame({"player_name": ["G", "D", "M", "F"], "team": list("ABCD"),
                           "position": ["GK", "DF", "MF", "FW"], "avg_rating": [7.0, 6.8, 7.2, 7.5]})
        assert (compute_ratings(df)["rank"] == 1).all()


# ============================================================
# Cas limites
# ============================================================

class TestEdgeCases:

    def test_empty_dataframe(self):
        empty = pd.DataFrame(columns=["player_name", "team", "position", "avg_rating"])
        result = compute_ratings(empty)
        assert isinstance(result, pd.DataFrame) and result.empty

    def test_missing_avg_rating_column_returns_empty(self):
        """API MPG indisponible (pas de note) : tableau vide, pas de plantage."""
        df = pd.DataFrame({"player_name": ["A"], "team": ["X"], "position": ["FW"], "total_goals": [3]})
        result = compute_ratings(df)
        assert result.empty and {"mpg_score", "rank"} <= set(result.columns)

    def test_all_players_filtered_by_games(self, mpg_players):
        assert len(compute_ratings(mpg_players, min_minutes=9999)) == 0


# ============================================================
# get_top_players() et get_value_picks()
# ============================================================

class TestSelections:

    def test_top_filter_by_position(self, mpg_players):
        top = get_top_players(compute_ratings(mpg_players), position="FW", min_games=0)
        assert len(top) > 0 and top["position"].str.startswith("FW").all()

    def test_top_n_respected(self, mpg_pool):
        rated = compute_ratings(mpg_pool)
        for n in [1, 5, 10]:
            assert len(get_top_players(rated, top_n=n, min_games=0)) <= n

    def test_top_min_games(self, mpg_players):
        top = get_top_players(compute_ratings(mpg_players), min_games=10)
        assert (top["games_played"] >= 10).all()

    def test_value_score_is_score_over_price(self, mpg_players):
        picks = get_value_picks(compute_ratings(mpg_players))
        assert np.allclose(picks["value_score"], (picks["mpg_score"] / picks["price"]).round(3))

    def test_value_picks_filters(self, mpg_players):
        """Prix d'au moins 5 M€, au moins 8 titularisations, budget max respecté."""
        rated = compute_ratings(mpg_players)
        picks = get_value_picks(rated, max_price=25)
        merged = picks.merge(rated[["player_name", "started_matches"]], on="player_name")
        assert (picks["price"].between(5, 25)).all()
        assert (merged["started_matches"] >= 8).all()

    def test_value_picks_sorted(self, mpg_pool):
        vs = get_value_picks(compute_ratings(mpg_pool), top_n=20)["value_score"].tolist()
        assert vs == sorted(vs, reverse=True)

    def test_value_picks_without_scores_returns_empty(self, mpg_players):
        assert get_value_picks(mpg_players).empty


# ============================================================
# optimize_xi() : programmation linéaire (PuLP)
# ============================================================

class TestOptimizeXI:

    def test_eleven_players_one_goalkeeper(self, mpg_pool):
        res = optimize_xi(compute_ratings(mpg_pool), budget=500)
        players = res["players"]
        assert len(players) == 11
        assert (players["position"] == "GK").sum() == 1

    def test_budget_respected(self, mpg_pool):
        res = optimize_xi(compute_ratings(mpg_pool), budget=150)
        assert res["total_price"] <= 150
        assert res["players"]["price"].sum() <= 150

    def test_formation_imposed(self, mpg_pool):
        res = optimize_xi(compute_ratings(mpg_pool), budget=500, formation=(4, 4, 2))
        counts = res["players"]["position"].str[:2].value_counts()
        assert res["formation"] == "4-4-2"
        assert (counts["DF"], counts["MF"], counts["FW"]) == (4, 4, 2)

    def test_max_three_per_club(self, mpg_pool):
        res = optimize_xi(compute_ratings(mpg_pool), budget=500)
        assert res["players"]["team"].value_counts().max() <= 3

    def test_locked_and_excluded_players(self, mpg_pool):
        rated = compute_ratings(mpg_pool)
        best = rated.iloc[0]["player_name"]
        weak = rated[rated["position"] == "FW"].iloc[-1]["player_name"]
        res = optimize_xi(rated, budget=500, locked_players=[weak], excluded_players=[best])
        names = set(res["players"]["player_name"])
        assert weak in names and best not in names

    def test_impossible_budget_returns_empty(self, mpg_pool):
        assert optimize_xi(compute_ratings(mpg_pool), budget=5) == {}
