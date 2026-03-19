"""
src/data/collect.py
-------------------
Ce fichier s'occupe de RÉCUPÉRER les données dont on a besoin.
C'est la première étape du projet : sans données, pas d'analyse !

On a 3 sources principales :
  1. L'API officielle MPG  → notes des joueurs + cotations (prix)
  2. FBref                 → statistiques détaillées des matchs (buts, passes, etc.)
  3. Understat             → métriques avancées (xG, xA)

Vocabulaire utile :
  - xG (expected goals)   : probabilité qu'un tir finisse en but. Un joueur avec xG=10
                            aurait "dû" marquer 10 buts selon la qualité de ses occasions.
  - xA (expected assists) : même principe pour les passes décisives.
  - Ces métriques sont plus fiables que les vrais buts/passes car elles
    éliminent la chance : un attaquant qui rate 20 occasions faciles a un xG élevé
    mais peu de vrais buts — c'est utile à savoir pour prédire ses futures perfs !
"""

# ============================================================
# IMPORTS : on charge les bibliothèques externes dont on a besoin
# ============================================================

import pandas as pd        # La bibliothèque principale pour manipuler des tableaux de données
import requests            # Pour faire des requêtes HTTP (appeler des APIs, des sites web)
import soccerdata as sd    # Bibliothèque spécialisée pour récupérer des stats de foot
from loguru import logger  # Pour afficher des messages joliment colorés dans le terminal
from pathlib import Path   # Pour gérer les chemins de fichiers de façon propre
import unicodedata         # Pour normaliser les accents dans les noms (é → e, etc.)

# ============================================================
# CONFIGURATION : les chemins et paramètres globaux du projet
# ============================================================

# Dossier où on va sauvegarder les données brutes téléchargées
# Path() crée un objet "chemin de fichier" compatible avec tous les systèmes d'exploitation
RAW_DIR = Path("data/raw")

# On crée le dossier s'il n'existe pas encore
# parents=True = crée aussi les dossiers parents si besoin
# exist_ok=True = ne génère pas d'erreur si le dossier existe déjà
RAW_DIR.mkdir(parents=True, exist_ok=True)

# URL de base de l'API officielle MPG
# Note : cette API est publique (non documentée officiellement mais utilisée
# par l'application MPG elle-même). Elle nous donne cotations et notes.
MPG_API_BASE = "https://api.mpg.football/api/data"

# Identifiant de la Ligue 1 dans le système MPG
# (chaque championnat a son propre code)
MPG_LEAGUE_ID = "mpg_league_season_2024_1"  # Ligue 1 saison 2024-2025


# ============================================================
# FONCTION 1 : Récupérer les notes et cotations MPG officielles
# ============================================================

def fetch_mpg_player_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Récupère les données officielles MPG pour tous les joueurs de Ligue 1.

    Ces données incluent :
      - La COTATION (prix) de chaque joueur en millions MPG
      - La NOTE MOYENNE sur la saison
      - Le poste du joueur selon MPG (GK, DF, MF, FW)
      - Le statut (disponible, blessé, suspendu)

    Pourquoi c'est précieux ?
      C'est la SOURCE DE VÉRITÉ pour MPG : ce sont exactement les prix et
      notes que tu vois dans l'application. Pas besoin de deviner !

    Args:
        force_refresh (bool): Si True, re-télécharge même si un fichier existe déjà.
                              Si False (défaut), utilise le fichier sauvegardé (plus rapide).

    Returns:
        pd.DataFrame: Un tableau avec une ligne par joueur.
    """
    # Chemin du fichier cache — pour ne pas re-télécharger à chaque exécution
    cache_path = RAW_DIR / "mpg_players.csv"

    # Si le fichier cache existe et qu'on ne force pas le rechargement, on le charge directement
    if cache_path.exists() and not force_refresh:
        logger.info("📂 Données MPG trouvées en cache, chargement...")
        return pd.read_csv(cache_path)

    logger.info("🌐 Téléchargement des données MPG officielles...")

    try:
        # On construit l'URL de l'API MPG pour les joueurs de Ligue 1
        url = f"{MPG_API_BASE}/championship-players-pool/{MPG_LEAGUE_ID}"

        # headers = informations qu'on envoie pour se "présenter" au serveur
        # User-Agent simule un navigateur web pour éviter d'être bloqué
        headers = {"User-Agent": "Mozilla/5.0 (compatible; MPGOptimizer/1.0)"}

        # On envoie la requête GET (comme ouvrir une page web)
        # timeout=15 : on abandonne après 15 secondes si pas de réponse
        response = requests.get(url, headers=headers, timeout=15)

        # .raise_for_status() génère une erreur Python si le serveur renvoie une erreur HTTP
        # (ex: 404 = page non trouvée, 500 = erreur serveur)
        response.raise_for_status()

        # .json() convertit la réponse texte (JSON) en dictionnaire Python
        # JSON est un format de données texte très utilisé dans les APIs
        data = response.json()

        # On extrait la liste des joueurs depuis la réponse JSON
        # .get("poolPlayers", []) retourne [] si la clé n'existe pas (évite une KeyError)
        players_raw = data.get("poolPlayers", [])

        if not players_raw:
            logger.warning("⚠️ L'API MPG a renvoyé une liste vide. Vérifiez l'URL ou la saison.")
            return pd.DataFrame()  # Retourne un tableau vide

        # On transforme la liste de dictionnaires JSON en DataFrame pandas
        # Chaque dictionnaire = un joueur ; chaque clé du dict = une colonne du tableau
        rows = []
        for p in players_raw:
            rows.append({
                "player_id":    p.get("id", ""),
                "player_name":  p.get("lastName", ""),
                "first_name":   p.get("firstName", ""),
                "team":         p.get("clubId", ""),
                # On convertit le code numérique de position en abréviation lisible
                "position":     _convert_mpg_position(p.get("ultraPosition", 0)),
                "price":        p.get("quotation", 0),          # Cotation MPG en millions
                "avg_rating":   p.get("averageRating", None),   # Note moyenne de saison
                "status":       _convert_mpg_status(p.get("injuryStatus", None)),
                # Les stats sont dans un sous-dictionnaire "stats"
                "games_played": p.get("stats", {}).get("matches", 0),
            })

        # pd.DataFrame() transforme notre liste de dictionnaires en tableau 2D
        df = pd.DataFrame(rows)

        # Sauvegarde en CSV pour les prochaines exécutions
        df.to_csv(cache_path, index=False)
        logger.success(f"✅ {len(df)} joueurs MPG récupérés et sauvegardés.")
        return df

    except requests.exceptions.RequestException as e:
        # Ce bloc s'exécute si la connexion a échoué (pas d'internet, API indisponible, etc.)
        logger.error(f"❌ Erreur réseau lors de l'appel API MPG : {e}")
        raise  # On "relance" l'erreur pour que l'appelant soit au courant


def fetch_mpg_ratings_history(force_refresh: bool = False) -> pd.DataFrame:
    """
    Récupère l'historique des notes MPG journée par journée.

    C'est très utile pour calculer la FORME RÉCENTE d'un joueur :
    un joueur avec des notes 7, 7.5, 8 sur les 3 dernières journées est en forme,
    même si sa moyenne de saison est 6 (il était blessé au début).

    Returns:
        pd.DataFrame: Tableau avec les colonnes player_id, journee, rating, goals, assists.
    """
    cache_path = RAW_DIR / "mpg_ratings_history.csv"

    if cache_path.exists() and not force_refresh:
        logger.info("📂 Historique des notes MPG trouvé en cache.")
        return pd.read_csv(cache_path)

    logger.info("🌐 Téléchargement de l'historique des notes MPG...")

    try:
        url = f"{MPG_API_BASE}/championship-stats-player/{MPG_LEAGUE_ID}"
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()

        rows = []
        # On parcourt chaque joueur dans la réponse JSON
        # .items() retourne des paires (clé, valeur) d'un dictionnaire
        for player_id, player_data in data.get("players", {}).items():
            # Pour chaque journée où il a joué
            for match_data in player_data.get("matches", []):
                rows.append({
                    "player_id": player_id,
                    "journee":   match_data.get("matchday", 0),
                    "rating":    match_data.get("rating", None),      # Note MPG de la journée
                    "goals":     match_data.get("goals", 0),
                    "assists":   match_data.get("assists", 0),
                    "minutes":   match_data.get("minutesPlayed", 0),
                })

        df = pd.DataFrame(rows)
        df.to_csv(cache_path, index=False)
        logger.success(f"✅ Historique récupéré : {len(df)} entrées joueur×journée.")
        return df

    except Exception as e:
        logger.error(f"❌ Erreur récupération historique notes : {e}")
        raise


# ============================================================
# FONCTION 2 : Récupérer les stats avancées depuis FBref
# ============================================================

def fetch_ligue1_stats(season: int = 2025, force_refresh: bool = False) -> pd.DataFrame:
    """
    Récupère les statistiques détaillées de Ligue 1 depuis FBref.

    FBref est une base de données football gratuite et très complète.
    Elle donne des stats que l'API MPG ne fournit pas directement :
      - xG (expected goals) et xA (expected assists)
      - Tirs, tirs cadrés
      - Passes progressives (qui avancent vers le but adverse)
      - Duels défensifs

    Args:
        season (int): Année de FIN de saison. Ex: 2025 pour la saison 2024-2025.
        force_refresh (bool): True = re-télécharge même si les données existent.

    Returns:
        pd.DataFrame: Stats détaillées par joueur pour la saison.
    """
    cache_path = RAW_DIR / f"fbref_stats_{season}.csv"

    if cache_path.exists() and not force_refresh:
        logger.info(f"📂 Stats FBref {season} trouvées en cache.")
        return pd.read_csv(cache_path)

    logger.info(f"🌐 Téléchargement des stats FBref saison {season-1}-{season}...")

    try:
        # La bibliothèque soccerdata s'occupe de tout le scraping de FBref
        fbref = sd.FBref(leagues="Ligue 1", seasons=season)

        # On récupère 3 types de stats différents depuis FBref
        # Chaque appel génère un tableau ; on les fusionnera ensuite

        # "standard" = statistiques principales : buts, passes décisives, minutes jouées...
        stats_standard = fbref.read_player_season_stats(stat_type="standard")

        # "shooting" = statistiques de tir : xG, nombre de tirs, tirs cadrés...
        stats_shooting = fbref.read_player_season_stats(stat_type="shooting")

        # "passing" = statistiques de passe : xA, passes clés, passes progressives...
        stats_passing = fbref.read_player_season_stats(stat_type="passing")

        # On fusionne les 3 tableaux sur les colonnes "player" et "team"
        # merge() est l'équivalent d'un VLOOKUP/RECHERCHEV, ou d'un JOIN SQL
        # how="left" signifie : garde tous les joueurs du tableau de gauche,
        # même s'ils n'ont pas de correspondance dans le tableau de droite
        df = (
            stats_standard
            .merge(
                # On sélectionne seulement les colonnes utiles de shooting
                # pour éviter d'avoir des colonnes en double (player, team déjà présents)
                stats_shooting[["player", "team", "xG", "npxG", "shots", "shots_on_target"]],
                on=["player", "team"],
                how="left"
            )
            .merge(
                stats_passing[["player", "team", "xAG", "key_passes", "progressive_passes"]],
                on=["player", "team"],
                how="left"
            )
        )

        df.to_csv(cache_path, index=False)
        logger.success(f"✅ {len(df)} joueurs FBref récupérés.")
        return df

    except Exception as e:
        logger.error(f"❌ Erreur lors de la collecte FBref : {e}")
        raise


# ============================================================
# FONCTION PRINCIPALE : Croiser toutes les sources en un seul tableau
# ============================================================

def build_master_dataset(season: int = 2025, force_refresh: bool = False) -> pd.DataFrame:
    """
    Construit le tableau FINAL en croisant MPG + FBref.

    C'est la fonction principale que tu utiliseras dans tes analyses.
    Elle appelle les fonctions ci-dessus et assemble tout proprement.

    Pourquoi croiser les sources ?
      - MPG seul : donne les prix et les notes officielles, mais peu de stats détaillées
      - FBref seul : donne les stats détaillées (xG, tirs...) mais pas les prix MPG
      - En combinant : on a une vue complète pour faire de bonnes décisions

    Schéma simplifié :
        Données MPG (prix, notes) ─── jointure sur le nom ──▶ Données FBref (xG, stats...)
                                                                    = Dataset Master complet

    Returns:
        pd.DataFrame: Dataset complet avec toutes les infos par joueur.
    """
    cache_path = RAW_DIR / f"master_dataset_{season}.csv"

    if cache_path.exists() and not force_refresh:
        logger.info("📂 Dataset master trouvé en cache.")
        return pd.read_csv(cache_path)

    logger.info("🔧 Construction du dataset master (MPG + FBref)...")

    # --- Étape 1 : Récupérer chaque source de données ---
    df_mpg   = fetch_mpg_player_data(force_refresh)
    df_fbref = fetch_ligue1_stats(season, force_refresh)

    # Si l'API MPG est indisponible, on continue avec FBref seul
    if df_mpg.empty:
        logger.warning("⚠️ Données MPG vides. Dataset basé sur FBref uniquement.")
        return df_fbref

    # --- Étape 2 : Normaliser les noms pour le croisement ---
    # Les noms peuvent différer entre les sources :
    #   MPG : "LACAZETTE"  vs  FBref : "Alexandre Lacazette"
    # La normalisation les ramène à un format commun pour faciliter la correspondance
    df_mpg["name_normalized"]   = df_mpg["player_name"].apply(_normalize_name)
    df_fbref["name_normalized"] = df_fbref["player"].apply(_normalize_name)

    # --- Étape 3 : Fusionner MPG et FBref ---
    df_master = df_mpg.merge(
        # On retire la colonne "player" de FBref pour éviter les doublons avec "player_name" de MPG
        df_fbref.drop(columns=["player"], errors="ignore"),
        on="name_normalized",
        how="left",
        # Si une colonne existe dans les deux tables, on ajoute un suffixe pour les distinguer
        suffixes=("_mpg", "_fbref")
    )

    # --- Étape 4 : Ajouter la forme récente (notes des 5 dernières journées) ---
    df_history = fetch_mpg_ratings_history(force_refresh)
    if not df_history.empty:
        # On calcule la moyenne des 5 dernières journées pour chaque joueur
        recent_form = (
            df_history
            .sort_values("journee", ascending=False)        # Du plus récent au plus ancien
            .groupby("player_id")                           # On regroupe par joueur
            .head(5)                                        # On ne garde que les 5 dernières lignes
            .groupby("player_id")["rating"]                 # On isole la colonne "rating"
            .mean()                                         # Moyenne
            .reset_index()                                  # Remet player_id en colonne normale
            .rename(columns={"rating": "recent_form_avg"}) # Renomme clairement
        )
        # On ajoute cette colonne au dataset master
        df_master = df_master.merge(recent_form, on="player_id", how="left")

    # --- Étape 5 : Sauvegarder ---
    df_master.to_csv(cache_path, index=False)
    logger.success(
        f"✅ Dataset master créé : {len(df_master)} joueurs, "
        f"{len(df_master.columns)} colonnes."
    )

    # Petit diagnostic : combien de joueurs n'ont pas de correspondance FBref ?
    if "xG" in df_master.columns:
        n_missing = df_master["xG"].isna().sum()
        logger.info(f"ℹ️  {n_missing} joueurs sans correspondance FBref (noms difficiles à matcher).")

    return df_master


# ============================================================
# FONCTIONS UTILITAIRES (internes — préfixées par _ par convention)
# Les fonctions avec _ ne sont pas censées être appelées de l'extérieur
# ============================================================

def _convert_mpg_position(ultra_position_code: int) -> str:
    """
    Convertit le code numérique de position MPG en abréviation lisible.

    MPG utilise des codes entiers pour les postes :
      10 = Gardien
      20 = Défenseur latéral, 21 = Défenseur central
      30 = Milieu défensif,   31 = Milieu offensif
      40 = Attaquant

    Args:
        ultra_position_code (int): Code numérique de poste MPG.

    Returns:
        str: 'GK', 'DF', 'MF', 'FW', ou '?' si code inconnu.
    """
    mapping = {
        10: "GK",   # Goalkeeper = Gardien
        20: "DF",   # Defender latéral
        21: "DF",   # Defender central
        30: "MF",   # Midfielder défensif
        31: "MF",   # Midfielder offensif
        40: "FW",   # Forward = Attaquant
    }
    # .get(clé, valeur_par_défaut) — retourne '?' si le code n'est pas dans le dictionnaire
    return mapping.get(ultra_position_code, "?")


def _convert_mpg_status(injury_status: str | None) -> str:
    """
    Convertit le code de statut MPG en texte lisible.

    Args:
        injury_status: Code de statut MPG (ex: "I1", "S", "D", ou None)

    Returns:
        str: 'Disponible', 'Blessé', 'Suspendu' ou 'Incertain'
    """
    if injury_status is None:
        return "Disponible"  # Pas de code = disponible

    # Les codes commençant par "I" indiquent une blessure (Injury)
    if injury_status.startswith("I"):
        return "Blessé"
    elif injury_status == "S":   # S = Suspended
        return "Suspendu"
    elif injury_status == "D":   # D = Doubtful
        return "Incertain"
    else:
        return "Disponible"


def _normalize_name(name: str) -> str:
    """
    Normalise un nom de joueur pour faciliter la comparaison entre sources.

    Problème : "Lacazette" dans MPG peut s'appeler "Alexandre Lacazette" dans FBref.
    Solution : on met tout en minuscules et on supprime les accents.

    Exemples :
        "LACAZETTE"           → "lacazette"
        "Mbappé"              → "mbappe"
        "  Guendouzi  "       → "guendouzi"

    Args:
        name (str): Nom brut à normaliser.

    Returns:
        str: Nom normalisé en minuscules, sans accents, sans espaces superflus.
    """
    if not isinstance(name, str):
        return ""  # Protection contre les valeurs None ou numériques

    # unicodedata.normalize("NFKD") décompose les caractères accentués en deux parties :
    # le caractère de base + le diacritique (accent).
    # Exemple : "é" → "e" + accent aigu
    nfkd = unicodedata.normalize("NFKD", name)

    # On filtre les caractères : on garde tout sauf les diacritiques (catégorie "Mn")
    # "Mn" = Mark, Nonspacing = les accents, trémas, cédilles...
    no_accents = "".join(c for c in nfkd if not unicodedata.category(c) == "Mn")

    # On met en minuscules et on supprime les espaces en début et fin
    return no_accents.lower().strip()


# ============================================================
# POINT D'ENTRÉE : exécuté uniquement si on lance ce fichier directement
# (et non quand il est importé par un autre script)
# ============================================================

if __name__ == "__main__":
    # Ce bloc est un "test rapide" pour vérifier que le code fonctionne

    print("=" * 50)
    print("  Test de collecte des données MPG")
    print("=" * 50)

    # Test de l'API MPG
    df_mpg = fetch_mpg_player_data()

    if not df_mpg.empty:
        print(f"\n✅ Données MPG OK : {len(df_mpg)} joueurs")
        print(f"Colonnes : {list(df_mpg.columns)}\n")

        # Affiche les 5 attaquants les mieux cotés
        top_fw = df_mpg[df_mpg["position"] == "FW"].nlargest(5, "price")
        print("Top 5 attaquants les plus chers :")
        print(top_fw[["player_name", "team", "price", "avg_rating", "status"]].to_string(index=False))
    else:
        print("\n⚠️ API MPG indisponible ou en cours de maintenance.")
        print("Essayez plus tard ou vérifiez la connexion internet.")
