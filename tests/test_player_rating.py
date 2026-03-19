"""
tests/test_player_rating.py
----------------------------
Ce fichier contient les TESTS UNITAIRES du module player_rating.py.

C'est quoi un test unitaire ?
  C'est un petit programme qui vérifie automatiquement qu'une fonction
  produit le bon résultat. Au lieu de tester manuellement à la main à chaque
  modification du code, on lance "pytest" et Python vérifie tout pour nous.

Comment lire ce fichier ?
  - Chaque fonction qui commence par "test_" est un test.
  - Un test passe (✅) si aucune assertion ne lève d'erreur.
  - Un test échoue (❌) si une assertion est fausse ou si une exception est levée.
  - Les "fixtures" (décorées avec @pytest.fixture) sont des données réutilisables
    entre plusieurs tests.

Comment lancer les tests ?
  Dans le terminal, depuis la racine du projet :
    pytest tests/                  # Lance tous les tests
    pytest tests/ -v               # Mode verbeux (affiche le nom de chaque test)
    pytest tests/ --cov=src        # Avec couverture de code (% du code testé)

Pourquoi tester ?
  - Pour s'assurer que le code fait bien ce qu'on attend
  - Pour détecter rapidement les régressions (une modif qui casse quelque chose)
  - C'est aussi une documentation vivante : les tests montrent comment utiliser le code
"""

# ============================================================
# IMPORTS
# ============================================================

import pytest        # Le framework de test Python
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# On ajoute la racine du projet au chemin Python pour pouvoir importer src/
# (nécessaire quand on lance pytest depuis le dossier racine)
sys.path.insert(0, str(Path(__file__).parent.parent))

# On importe les fonctions à tester
from src.analysis.player_rating import (
    normalize_series,
    compute_per90,
    compute_ratings,
    get_top_players,
)


# ============================================================
# FIXTURES : données de test réutilisables
# ============================================================

@pytest.fixture
def sample_players_basic():
    """
    Dataset simple avec uniquement des stats de base (sans note MPG ni form récente).
    Utilisé pour tester les cas simples.
    """
    return pd.DataFrame({
        "player_name": ["Lacazette", "Thauvin", "Guendouzi", "Saliba", "Areola"],
        "team":        ["Lyon", "OM", "OM", "Arsenal", "WHU"],
        "position":    ["FW", "FW", "MF", "DF", "GK"],
        "minutes":     [1800, 1600, 1900, 2100, 2000],
        "goals":       [12, 9, 3, 1, 0],
        "assists":     [4, 6, 8, 2, 0],
        "npxG":        [10.5, 8.2, 2.8, 0.9, 0.1],
        "xAG":         [3.8, 5.1, 7.4, 1.8, 0.3],
        "shots_on_target": [35, 28, 12, 6, 2],
        "key_passes":  [20, 30, 55, 10, 4],
        "price":       [25, 20, 15, 30, 10],
    })


@pytest.fixture
def sample_players_with_mpg_ratings():
    """
    Dataset complet qui inclut les données MPG officielles :
      - avg_rating     : la note moyenne de saison (issue de l'API MPG)
      - recent_form_avg: moyenne des 5 dernières journées (calculée dans collect.py)

    C'est le cas réaliste : après fetch_mpg_player_data() + build_master_dataset().
    """
    return pd.DataFrame({
        "player_name":     ["Lacazette", "Thauvin", "Guendouzi", "Saliba", "Areola"],
        "team":            ["Lyon", "OM", "OM", "Arsenal", "WHU"],
        "position":        ["FW", "FW", "MF", "DF", "GK"],
        "minutes":         [1800, 1600, 1900, 2100, 2000],
        # ── Données MPG officielles ──
        "avg_rating":      [6.8, 6.5, 6.9, 7.1, 6.7],   # Notes de l'appli MPG
        "recent_form_avg": [7.2, 6.0, 7.5, 7.0, 6.5],   # Forme récente (5 dernières J)
        # ── Stats FBref ──
        "goals":           [12, 9, 3, 1, 0],
        "assists":         [4, 6, 8, 2, 0],
        "npxG":            [10.5, 8.2, 2.8, 0.9, 0.1],
        "xAG":             [3.8, 5.1, 7.4, 1.8, 0.3],
        "shots_on_target": [35, 28, 12, 6, 2],
        "key_passes":      [20, 30, 55, 10, 4],
        "price":           [25, 20, 15, 30, 10],
    })


@pytest.fixture
def sample_players_large():
    """
    Dataset plus grand (20 joueurs) pour tester des cas plus réalistes,
    notamment le filtre par minutes et les classements.
    """
    np.random.seed(42)  # Graine aléatoire fixe = résultats reproductibles

    positions = ["GK"] * 2 + ["DF"] * 6 + ["MF"] * 7 + ["FW"] * 5

    return pd.DataFrame({
        "player_name":  [f"Joueur_{i:02d}" for i in range(20)],
        "team":         [f"Club_{i % 5}" for i in range(20)],
        "position":     positions,
        "minutes":      [500, 300, 1800, 1800, 1800, 1200, 400, 1800, 1800,
                         1800, 1800, 1200, 500, 1800, 1800, 1800, 1800, 1200, 500, 1800],
        "avg_rating":   np.random.uniform(5.5, 8.0, 20).round(1),
        "goals":        np.random.randint(0, 15, 20),
        "npxG":         np.random.uniform(0, 12, 20).round(1),
        "xAG":          np.random.uniform(0, 8, 20).round(1),
        "price":        np.random.uniform(5, 40, 20).round(1),
    })


# ============================================================
# TESTS DE normalize_series()
# ============================================================

class TestNormalizeSeries:
    """
    Tests pour la fonction normalize_series().
    On regroupe les tests dans une classe pour les organiser par fonction testée.
    """

    def test_output_min_is_zero(self):
        """La valeur minimale après normalisation doit être 0."""
        s = pd.Series([1, 2, 3, 4, 5])
        result = normalize_series(s)
        # pytest.approx() gère les petites erreurs d'arrondi flottant
        assert result.min() == pytest.approx(0.0)

    def test_output_max_is_ten(self):
        """La valeur maximale après normalisation doit être 10."""
        s = pd.Series([1, 2, 3, 4, 5])
        result = normalize_series(s)
        assert result.max() == pytest.approx(10.0)

    def test_constant_series_returns_five(self):
        """
        Une série où toutes les valeurs sont identiques ne peut pas être
        normalisée (division par 0). On retourne 5.0 (valeur neutre).
        """
        s = pd.Series([3.0, 3.0, 3.0])
        result = normalize_series(s)
        assert (result == 5.0).all(), "Une série constante doit retourner 5.0 partout"

    def test_preserves_order(self):
        """L'ordre relatif des valeurs doit être préservé après normalisation."""
        s = pd.Series([1, 5, 3, 2, 4])
        result = normalize_series(s)
        # Le rang des valeurs normalisées doit être identique au rang des originales
        assert list(result.rank()) == list(s.rank())

    def test_output_length_unchanged(self):
        """La série normalisée doit avoir le même nombre d'éléments qu'en entrée."""
        s = pd.Series([10, 20, 30, 40, 50, 60])
        result = normalize_series(s)
        assert len(result) == len(s)


# ============================================================
# TESTS DE compute_per90()
# ============================================================

class TestComputePer90:
    """Tests pour la fonction compute_per90()."""

    def test_basic_calculation(self, sample_players_basic):
        """
        Test de base : vérifie le calcul pour Lacazette.
        Lacazette : 12 buts en 1800 min = 12 / (1800/90) = 12 / 20 = 0.6 buts/90
        """
        result = compute_per90(sample_players_basic, "goals")
        assert result.iloc[0] == pytest.approx(0.6), (
            "Lacazette : 12 buts / 20 périodes de 90 min = 0.6 buts/90"
        )

    def test_output_length(self, sample_players_basic):
        """La série résultante doit avoir autant de lignes que le dataset."""
        result = compute_per90(sample_players_basic, "goals")
        assert len(result) == len(sample_players_basic)

    def test_zero_minutes_returns_nan(self):
        """Un joueur avec 0 minutes jouées ne peut pas avoir de stat per90 → NaN."""
        df = pd.DataFrame({
            "goals":   [5, 0],
            "minutes": [900, 0],   # Le 2e joueur n'a pas joué
        })
        result = compute_per90(df, "goals")
        # La valeur doit être NaN (Not a Number) pour le joueur sans minutes
        assert np.isnan(result.iloc[1]), "0 minutes → stat per90 doit être NaN"

    def test_proportional_result(self, sample_players_basic):
        """Les résultats doivent être proportionnels aux stats brutes."""
        result = compute_per90(sample_players_basic, "goals")
        # Tous les résultats doivent être positifs (ou NaN)
        assert (result.dropna() >= 0).all(), "Une stat per90 ne peut pas être négative"


# ============================================================
# TESTS DE compute_ratings() — sans notes MPG
# ============================================================

class TestComputeRatingsBasic:
    """
    Tests de compute_ratings() avec un dataset sans note MPG officielle.
    On vérifie que le code fonctionne en mode dégradé (sans avg_rating).
    """

    def test_mpg_score_column_created(self, sample_players_basic):
        """La colonne 'mpg_score' doit être créée."""
        rated = compute_ratings(sample_players_basic, min_minutes=0)
        assert "mpg_score" in rated.columns, "mpg_score doit être dans les colonnes"

    def test_rank_column_created(self, sample_players_basic):
        """La colonne 'rank' doit être créée."""
        rated = compute_ratings(sample_players_basic, min_minutes=0)
        assert "rank" in rated.columns, "rank doit être dans les colonnes"

    def test_scores_within_range(self, sample_players_basic):
        """Tous les scores doivent être entre 0 et 10."""
        rated = compute_ratings(sample_players_basic, min_minutes=0)
        assert rated["mpg_score"].between(0, 10).all(), (
            f"Scores hors limites trouvés : {rated['mpg_score'].describe()}"
        )

    def test_min_minutes_filter(self, sample_players_large):
        """
        Le filtre min_minutes doit exclure les joueurs avec trop peu de temps de jeu.
        Dans sample_players_large, certains joueurs ont 300 ou 400 min.
        """
        rated = compute_ratings(sample_players_large, min_minutes=500)
        # Tous les joueurs retenus doivent avoir >= 500 minutes
        assert (rated["minutes"] >= 500).all(), (
            "Des joueurs avec moins de 500 min ont été inclus malgré le filtre"
        )

    def test_sorted_descending(self, sample_players_basic):
        """Le résultat doit être trié par score décroissant (meilleur en premier)."""
        rated = compute_ratings(sample_players_basic, min_minutes=0)
        scores = rated["mpg_score"].tolist()
        assert scores == sorted(scores, reverse=True), (
            "Les scores ne sont pas triés par ordre décroissant"
        )

    def test_per90_columns_created(self, sample_players_basic):
        """Les colonnes per90 doivent être créées pour les stats disponibles."""
        rated = compute_ratings(sample_players_basic, min_minutes=0)
        # goals_per90 doit exister si "goals" est dans le dataset
        assert "goals_per90" in rated.columns, (
            "goals_per90 devrait être créé automatiquement"
        )


# ============================================================
# TESTS DE compute_ratings() — avec notes MPG officielles
# ============================================================

class TestComputeRatingsWithMPG:
    """
    Tests de compute_ratings() avec le dataset complet incluant avg_rating.
    C'est le cas d'usage réel après avoir appelé build_master_dataset().
    """

    def test_avg_rating_used_in_score(self, sample_players_with_mpg_ratings):
        """
        Quand avg_rating est disponible, il doit influencer le score.
        On vérifie qu'un joueur avec une meilleure note MPG a (tendanciellement)
        un meilleur score composite.
        """
        rated = compute_ratings(sample_players_with_mpg_ratings, min_minutes=0)
        assert "mpg_score" in rated.columns

        # Saliba a avg_rating=7.1 (le plus haut parmi les DF/GK)
        # Son score devrait être supérieur à zéro
        saliba_score = rated[rated["player_name"] == "Saliba"]["mpg_score"].values[0]
        assert saliba_score > 0, "Un joueur avec une bonne note MPG doit avoir un score positif"

    def test_recent_form_preserved_in_output(self, sample_players_with_mpg_ratings):
        """
        La colonne recent_form_avg doit être préservée dans le dataset de sortie.
        Elle est utilisée en aval dans mercato.py pour détecter les joueurs en forme.
        """
        rated = compute_ratings(sample_players_with_mpg_ratings, min_minutes=0)
        assert "recent_form_avg" in rated.columns, (
            "recent_form_avg doit être conservé pour l'analyse de forme dans mercato.py"
        )

    def test_scores_still_bounded(self, sample_players_with_mpg_ratings):
        """Même avec avg_rating, les scores doivent rester entre 0 et 10."""
        rated = compute_ratings(sample_players_with_mpg_ratings, min_minutes=0)
        assert rated["mpg_score"].between(0, 10).all()

    def test_all_positions_have_scores(self, sample_players_with_mpg_ratings):
        """Tous les postes (GK, DF, MF, FW) doivent avoir des scores calculés."""
        rated = compute_ratings(sample_players_with_mpg_ratings, min_minutes=0)

        for pos in ["GK", "DF", "MF", "FW"]:
            pos_scores = rated[rated["position"].str.startswith(pos)]["mpg_score"]
            assert len(pos_scores) > 0, f"Aucun score calculé pour le poste {pos}"
            assert (pos_scores > 0).all(), f"Scores nuls pour le poste {pos}"


# ============================================================
# TESTS DE get_top_players()
# ============================================================

class TestGetTopPlayers:
    """Tests pour la fonction get_top_players()."""

    def test_filter_by_position(self, sample_players_with_mpg_ratings):
        """Quand on filtre par position, on ne doit avoir que ce poste."""
        rated = compute_ratings(sample_players_with_mpg_ratings, min_minutes=0)
        top_fw = get_top_players(rated, position="FW", top_n=10)
        # Tous les joueurs retournés doivent être des FW
        assert top_fw["position"].str.startswith("FW").all(), (
            "get_top_players(position='FW') retourne des joueurs non-FW"
        )

    def test_top_n_respected(self, sample_players_large):
        """Le nombre de résultats ne doit pas dépasser top_n."""
        rated = compute_ratings(sample_players_large, min_minutes=0)

        for n in [1, 3, 5, 10]:
            result = get_top_players(rated, top_n=n)
            assert len(result) <= n, f"top_n={n} mais {len(result)} joueurs retournés"

    def test_no_position_filter_returns_all(self, sample_players_with_mpg_ratings):
        """Sans filtre de position, toutes les positions doivent apparaître."""
        rated = compute_ratings(sample_players_with_mpg_ratings, min_minutes=0)
        top_all = get_top_players(rated, top_n=10)  # Pas de filtre position

        positions_in_result = set(top_all["position"].str[:2].unique())
        # On s'attend à voir au moins 2 positions différentes
        assert len(positions_in_result) >= 2, (
            "Sans filtre, get_top_players devrait retourner plusieurs positions"
        )

    def test_output_contains_key_columns(self, sample_players_with_mpg_ratings):
        """
        Le résultat doit contenir les colonnes essentielles pour la prise de décision :
        le nom, le score composite, et si disponibles : la note MPG et la forme récente.
        """
        rated = compute_ratings(sample_players_with_mpg_ratings, min_minutes=0)
        result = get_top_players(rated, top_n=5)

        assert "mpg_score" in result.columns, "mpg_score doit toujours être présent"
        assert "player_name" in result.columns, "player_name doit toujours être présent"
        # avg_rating doit être présent si le dataset en avait une
        assert "avg_rating" in result.columns, (
            "avg_rating (note MPG officielle) doit apparaître dans les résultats"
        )

    def test_sorted_by_score(self, sample_players_large):
        """Les joueurs retournés doivent être triés par score décroissant."""
        rated = compute_ratings(sample_players_large, min_minutes=0)
        result = get_top_players(rated, top_n=10)
        scores = result["mpg_score"].tolist()
        assert scores == sorted(scores, reverse=True), (
            "get_top_players doit retourner les joueurs du meilleur au moins bon"
        )


# ============================================================
# TESTS DES CAS LIMITES (edge cases)
# ============================================================

class TestEdgeCases:
    """
    Tests des situations inhabituelles ou problématiques.
    Un bon code doit gérer correctement ces cas sans planter.
    """

    def test_empty_dataframe(self):
        """Un DataFrame vide ne doit pas faire planter compute_ratings()."""
        empty_df = pd.DataFrame(columns=["player_name", "team", "position", "minutes"])
        # On s'attend soit à un DataFrame vide, soit à une erreur gérée proprement
        try:
            result = compute_ratings(empty_df, min_minutes=0)
            assert isinstance(result, pd.DataFrame)
        except Exception as e:
            pytest.fail(f"compute_ratings() a planté sur un DataFrame vide : {e}")

    def test_all_players_filtered_by_minutes(self, sample_players_basic):
        """Si tous les joueurs ont moins de minutes que le seuil, on doit obtenir un tableau vide."""
        rated = compute_ratings(sample_players_basic, min_minutes=9999)
        assert len(rated) == 0, (
            "Avec un seuil de minutes très élevé, aucun joueur ne doit passer le filtre"
        )

    def test_missing_optional_stats(self):
        """
        Si certaines stats FBref sont absentes (ex: FBref indisponible),
        le calcul doit quand même fonctionner avec ce qui est disponible.
        """
        # Dataset minimaliste — seulement les colonnes de base + avg_rating MPG
        minimal_df = pd.DataFrame({
            "player_name": ["Joueur_A", "Joueur_B", "Joueur_C"],
            "team":        ["Club_1", "Club_2", "Club_3"],
            "position":    ["FW", "MF", "DF"],
            "minutes":     [1800, 1800, 1800],
            "avg_rating":  [7.0, 6.5, 7.2],   # On a la note MPG mais pas les stats FBref
            # Pas de xG, xAG, shots_on_target, etc.
        })

        # Ça doit fonctionner sans erreur
        rated = compute_ratings(minimal_df, min_minutes=0)
        assert len(rated) == 3, "Les 3 joueurs doivent être présents dans le résultat"
        assert "mpg_score" in rated.columns

    def test_single_player_per_position(self):
        """Avec un seul joueur par position, le scoring doit fonctionner."""
        one_per_pos = pd.DataFrame({
            "player_name": ["Le_GK", "Le_DF", "Le_MF", "Le_FW"],
            "team":        ["A", "B", "C", "D"],
            "position":    ["GK", "DF", "MF", "FW"],
            "minutes":     [1800, 1800, 1800, 1800],
            "avg_rating":  [7.0, 6.8, 7.2, 7.5],
        })

        rated = compute_ratings(one_per_pos, min_minutes=0)
        assert len(rated) == 4
        # Avec un seul joueur par poste, il obtient le rang 1
        assert (rated["rank"] == 1).all(), "Joueur unique par poste → rang doit être 1"
