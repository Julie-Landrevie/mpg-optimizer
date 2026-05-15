"""
src/data/collect.py
-------------------
Ce fichier s'occupe de RÉCUPÉRER les données dont on a besoin.
C'est la première étape du projet : sans données, pas d'analyse !

On a 3 sources principales :
  1. L'API officielle MPG  → notes des joueurs + cotations (prix) ✅ Fonctionne
  2. FBref via CSV         → statistiques avancées (xG, xA, etc.) ✅ Via fichier CSV
  3. Understat             → métriques encore plus précises (prévu plus tard)

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
# "1" = Ligue 1 (vérifié en explorant l'API championship-clubs)
MPG_LEAGUE_ID = "1"

# Nom du fichier CSV FBref à placer dans data/raw/
# Pour le générer : fbref.com/en/comps/13/stats/Ligue-1-Stats
# → "Share & more" → "Get table as CSV" → copier dans un fichier .csv
FBREF_CSV_FILENAME = "ligue1_stats_2025.csv"


# ============================================================
# FONCTION 1 : Récupérer les notes et cotations MPG officielles
# ============================================================

def fetch_mpg_player_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Récupère les données officielles MPG pour tous les joueurs de Ligue 1.

    Ces données incluent :
      - La COTATION (prix) de chaque joueur en millions MPG
      - La NOTE MOYENNE sur la saison
      - Les 5 DERNIÈRES NOTES journée par journée (forme récente !)
      - Le total de buts, clean sheets, cartons...
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
        # MPG_LEAGUE_ID = "1" correspond à la Ligue 1
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
            logger.warning("⚠️ L'API MPG a renvoyé une liste vide.")
            return pd.DataFrame()

        # On transforme la liste de dictionnaires JSON en DataFrame pandas
        # Chaque dictionnaire = un joueur ; chaque clé du dict = une colonne du tableau
        rows = []
        for p in players_raw:
            # Les statistiques sont dans un sous-dictionnaire "stats"
            stats = p.get("stats", {})

            rows.append({
                "player_id":       p.get("id", ""),
                "player_name":     p.get("lastName", ""),
                "first_name":      p.get("firstName", ""),
                "team":            p.get("clubId", ""),
                # On convertit le code numérique de position en abréviation lisible
                "position":        _convert_mpg_position(p.get("ultraPosition", 0)),
                "price":           p.get("quotation", 0),        # Cotation MPG en millions
                "avg_rating":      stats.get("averageRating", None),  # Note moyenne de saison
                "avg_points":      stats.get("averagePoints", None),  # Points MPG moyens
                "status":          _convert_mpg_status(p.get("injuryStatus", None)),
                "games_played":    stats.get("totalMatches", 0),
                "started_matches": stats.get("totalStartedMatches", 0),
                # Les 5 dernières notes : liste convertie en texte pour le CSV
                # Ex: [6.5, 7, 5.5, 8, 6] → on peut l'utiliser pour la forme récente
                "last_5_ratings":  str(stats.get("lastRatings", [])),
                # Stats détaillées
                "total_goals":     stats.get("totalGoals", 0),
                "clean_sheets":    stats.get("totalCleanSheets", 0),
                "goals_conceded":  stats.get("totalGoalsConceded", 0),
                "yellow_cards":    stats.get("totalYellowCards", 0),
                "red_cards":       stats.get("totalRedCards", 0),
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
        raise


def compute_recent_form(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la moyenne des 5 dernières notes MPG pour chaque joueur.

    Les 5 dernières notes sont stockées dans la colonne 'last_5_ratings'
    sous forme de texte (ex: "[6.5, 7.0, 5.5, 8.0, 6.0]").
    On les convertit en liste Python et on calcule la moyenne.

    Cette fonction est un enrichissement de fetch_mpg_player_data() :
    on l'appelle après pour ajouter la colonne 'recent_form_avg'.

    Args:
        df (pd.DataFrame): DataFrame retourné par fetch_mpg_player_data().

    Returns:
        pd.DataFrame: Même DataFrame avec la colonne 'recent_form_avg' ajoutée.
    """
    import ast  # ast = Abstract Syntax Tree, permet de convertir du texte en liste Python

    def parse_ratings(ratings_str):
        """Convertit le texte '[6.5, 7.0, 5.5]' en moyenne numérique."""
        try:
            # ast.literal_eval convertit une chaîne comme "[6.5, 7.0]" en liste Python [6.5, 7.0]
            # C'est plus sûr que eval() car il n'exécute pas de code arbitraire
            ratings = ast.literal_eval(ratings_str)
            # On filtre les None et les valeurs non numériques (certains joueurs n'ont pas joué)
            ratings = [r for r in ratings if isinstance(r, (int, float))]
            if ratings:  # Si la liste n'est pas vide après filtrage
                return round(sum(ratings) / len(ratings), 2)
        except (ValueError, SyntaxError):
            pass
        return None  # Retourne None si impossible à parser

    df = df.copy()
    df["recent_form_avg"] = df["last_5_ratings"].apply(parse_ratings)
    return df


# ============================================================
# FONCTION 2 : Charger les stats FBref depuis le fichier CSV
# ============================================================

def load_fbref_csv(filename: str = FBREF_CSV_FILENAME) -> pd.DataFrame:
    """
    Charge les statistiques FBref depuis un fichier CSV téléchargé manuellement.

    Pourquoi CSV et pas scraping automatique ?
      FBref bloque les requêtes automatiques (erreur 403 Forbidden).
      La solution propre est de télécharger les données manuellement depuis leur site.

    Comment obtenir le fichier CSV ?
      1. Va sur https://fbref.com/en/comps/13/stats/Ligue-1-Stats
      2. Fais défiler jusqu'au tableau des statistiques joueurs
      3. Clique sur "Share & more" (icône partage) → "Get table as CSV"
      4. Copie tout le texte affiché
      5. Colle-le dans un fichier texte nommé 'ligue1_stats_2025.csv'
      6. Place ce fichier dans le dossier data/raw/

    Note pour Mac sans Excel :
      Tu peux créer ce fichier dans Numbers ou même TextEdit en mode texte brut.
      L'important c'est que le fichier s'appelle exactement 'ligue1_stats_2025.csv'
      et qu'il soit dans data/raw/.

    Args:
        filename (str): Nom du fichier CSV dans data/raw/. Défaut = 'ligue1_stats_2025.csv'

    Returns:
        pd.DataFrame: Stats FBref, ou DataFrame vide si fichier introuvable.
    """
    csv_path = RAW_DIR / filename

    if not csv_path.exists():
        logger.warning(
            f"⚠️ Fichier FBref non trouvé : {csv_path}\n"
            f"   → Va sur fbref.com/en/comps/13/stats/Ligue-1-Stats\n"
            f"   → 'Share & more' → 'Get table as CSV'\n"
            f"   → Sauvegarde le fichier dans data/raw/{filename}"
        )
        return pd.DataFrame()

    logger.info(f"📂 Chargement du fichier FBref : {csv_path}")

    try:
        # On essaie de lire le CSV — FBref peut avoir des lignes d'en-tête multiples
        # skiprows=1 permet de sauter une éventuelle première ligne parasite
        df = pd.read_csv(csv_path, skiprows=1, sep=";", encoding="utf-8", lineterminator="\n", on_bad_lines="skip")

        # Nettoyage basique : supprimer les lignes où le joueur s'appelle "Player"
        # (FBref répète parfois les en-têtes au milieu du tableau)
        if "Player" in df.columns:
            df = df[df["Player"] != "Player"]
            df = df.rename(columns={"Player": "player"})

        # Supprimer les colonnes entièrement vides
        df = df.dropna(axis=1, how="all")

        # Convertir les colonnes numériques
        for col in df.columns:
            if col not in ["player", "Nation", "Pos", "Squad", "Comp", "Age", "Born"]:
                df[col] = pd.to_numeric(df[col], errors="ignore")

        logger.success(f"✅ FBref chargé : {len(df)} joueurs, {len(df.columns)} colonnes.")
        return df

    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement du CSV FBref : {e}")
        return pd.DataFrame()


# ============================================================
# FONCTION PRINCIPALE : Construire le dataset complet
# ============================================================

def build_master_dataset(force_refresh: bool = False) -> pd.DataFrame:
    """
    Construit le tableau FINAL en croisant MPG + FBref (si disponible).

    C'est la fonction principale que tu utiliseras dans tes analyses.
    Elle appelle les fonctions ci-dessus et assemble tout proprement.

    Pourquoi croiser les sources ?
      - MPG seul : donne les prix et les notes officielles + last_5_ratings
      - FBref seul : donne les stats avancées (xG, tirs...) mais pas les prix MPG
      - En combinant : on a une vue complète pour faire de bonnes décisions

    Returns:
        pd.DataFrame: Dataset complet avec toutes les infos par joueur.
    """
    cache_path = RAW_DIR / "master_dataset.csv"

    if cache_path.exists() and not force_refresh:
        logger.info("📂 Dataset master trouvé en cache.")
        return pd.read_csv(cache_path)

    logger.info("🔧 Construction du dataset master...")

    # --- Étape 1 : Données MPG (obligatoires) ---
    df_mpg = fetch_mpg_player_data(force_refresh)

    if df_mpg.empty:
        logger.error("❌ Impossible de construire le dataset sans données MPG.")
        return pd.DataFrame()

    # Calcul de la forme récente à partir des 5 dernières notes
    df_mpg = compute_recent_form(df_mpg)

    # --- Étape 2 : Données FBref (optionnelles) ---
    df_fbref = load_fbref_csv()

    if df_fbref.empty:
        logger.warning("⚠️ FBref non disponible. Dataset basé sur MPG uniquement.")
        df_mpg.to_csv(cache_path, index=False)
        return df_mpg

    # --- Étape 3 : Normaliser les noms ---
    # Problème : MPG stocke uniquement le NOM DE FAMILLE ("Abergel")
    # alors que FBref a PRÉNOM + NOM ("Laurent Abergel").
    # Solution : matching en 3 passes, du plus précis au moins précis.

    fbref_name_col = "player" if "player" in df_fbref.columns else "Player"

    # Clés FBref
    df_fbref["name_normalized"]      = df_fbref[fbref_name_col].apply(_normalize_name)
    df_fbref["last_name_normalized"] = df_fbref["name_normalized"].apply(
        lambda n: n.split()[-1] if n else ""
    )

    # Clés MPG
    df_mpg["full_name_normalized"] = (
        df_mpg["first_name"].fillna("").apply(_normalize_name) + " " +
        df_mpg["player_name"].apply(_normalize_name)
    ).str.strip()
    df_mpg["name_normalized"] = df_mpg["player_name"].apply(_normalize_name)

    fbref_drop  = [fbref_name_col]
    fbref_cols  = [c for c in df_fbref.columns if c not in fbref_drop + ["name_normalized", "last_name_normalized"]]
    stat_col    = "Gls" if "Gls" in df_fbref.columns else (fbref_cols[0] if fbref_cols else None)

    def is_matched(df_m):
        if stat_col and stat_col in df_m.columns:
            return df_m[stat_col].notna()
        return pd.Series([False] * len(df_m), index=df_m.index)

    # PASSE 1 : prénom+nom exact ("laurent abergel" == "laurent abergel")
    merge1    = df_mpg.merge(
        df_fbref[["name_normalized"] + fbref_cols],
        left_on="full_name_normalized", right_on="name_normalized",
        how="left", suffixes=("", "_f1")
    )
    matched1  = is_matched(merge1)
    n_pass1   = matched1.sum()
    logger.info(f"Passe 1 (prénom+nom exact)   : {n_pass1} joueurs matchés")

    # PASSE 2 : nom seul, uniquement si nom unique dans FBref (évite faux positifs)
    fbref_uniq = df_fbref.drop_duplicates(subset="last_name_normalized", keep=False)
    idx2       = merge1[~matched1].index
    merge2     = df_mpg.loc[idx2].merge(
        fbref_uniq[["last_name_normalized"] + fbref_cols],
        left_on="name_normalized", right_on="last_name_normalized",
        how="left", suffixes=("", "_f2")
    )
    matched2   = is_matched(merge2)
    n_pass2    = matched2.sum()
    logger.info(f"Passe 2 (nom seul unique)    : {n_pass2} joueurs matchés")

    # PASSE 3 : dictionnaire lookup pour les cas restants
    fbref_dict = {row["last_name_normalized"]: row for _, row in df_fbref.iterrows()}
    idx3       = merge2[~matched2].index
    extra      = []
    n_pass3    = 0
    for i in idx3:
        r        = df_mpg.loc[i].to_dict()
        fbref_r  = fbref_dict.get(r.get("name_normalized", ""))
        if fbref_r is not None:
            for c in fbref_cols:
                r[c] = fbref_r.get(c)
            n_pass3 += 1
        extra.append(r)
    logger.info(f"Passe 3 (lookup dictionnaire): {n_pass3} joueurs matchés")

    # --- Assemblage ---
    # On repart du dataset MPG complet (514 joueurs) et on enrichit colonne par colonne
    # pour ne perdre aucun joueur.
    df_master = df_mpg.copy()

    # Colonnes FBref à ajouter
    for col in fbref_cols:
        df_master[col] = None

    # Appliquer les résultats de passe 1
    for idx, row in merge1[matched1].iterrows():
        pid = row.get("player_id")
        if pid:
            for col in fbref_cols:
                if col in row:
                    df_master.loc[df_master["player_id"] == pid, col] = row[col]

    # Appliquer les résultats de passe 2 (joueurs pas encore matchés)
    already_matched = set(merge1[matched1]["player_id"].tolist())
    for idx, row in merge2[matched2].iterrows():
        pid = row.get("player_id")
        if pid and pid not in already_matched:
            for col in fbref_cols:
                if col in row:
                    df_master.loc[df_master["player_id"] == pid, col] = row[col]
            already_matched.add(pid)

    # Appliquer les résultats de passe 3
    for r in extra:
        pid = r.get("player_id")
        if pid and pid not in already_matched and r.get("Gls") is not None:
            for col in fbref_cols:
                if col in r:
                    df_master.loc[df_master["player_id"] == pid, col] = r.get(col)
            already_matched.add(pid)

    total_matched = n_pass1 + n_pass2 + n_pass3

    # --- Étape 5 : Sauvegarde ---
    df_master.to_csv(cache_path, index=False)
    logger.success(
        f"✅ Dataset master : {len(df_master)} joueurs, "
        f"{len(df_master.columns)} colonnes. "
        f"{total_matched} joueurs croisés avec FBref (avant : 27) 🎯"
    )

    return df_master


# ============================================================
# FONCTIONS UTILITAIRES (internes — préfixées par _ par convention)
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
        return "Disponible"

    if injury_status.startswith("I"):
        return "Blessé"
    elif injury_status == "S":
        return "Suspendu"
    elif injury_status == "D":
        return "Incertain"
    else:
        return "Disponible"


def _normalize_name(name: str) -> str:
    """
    Normalise un nom de joueur pour faciliter la comparaison entre sources.

    Problème : "Lacazette" dans MPG peut s'appeler "Alexandre Lacazette" dans FBref.
    Solution : on met tout en minuscules et on supprime les accents.

    Exemples :
        "LACAZETTE"    → "lacazette"
        "Mbappé"       → "mbappe"
        "  Guendouzi " → "guendouzi"

    Args:
        name (str): Nom brut à normaliser.

    Returns:
        str: Nom normalisé en minuscules, sans accents, sans espaces superflus.
    """
    if not isinstance(name, str):
        return ""

    # unicodedata.normalize("NFKD") décompose les caractères accentués :
    # "é" → "e" + accent aigu (deux caractères séparés)
    nfkd = unicodedata.normalize("NFKD", name)

    # On garde tout sauf les diacritiques (catégorie "Mn" = accents, trémas...)
    no_accents = "".join(c for c in nfkd if not unicodedata.category(c) == "Mn")

    return no_accents.lower().strip()


# ============================================================
# POINT D'ENTRÉE — test rapide quand on lance ce fichier directement
# ============================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  Test de collecte des données MPG Optimizer")
    print("=" * 55)

    # --- Test 1 : API MPG ---
    print("\n📡 Test de l'API MPG...")
    df_mpg = fetch_mpg_player_data()

    if not df_mpg.empty:
        # Calcul de la forme récente
        df_mpg = compute_recent_form(df_mpg)

        print(f"\n✅ API MPG OK : {len(df_mpg)} joueurs récupérés")
        print(f"Colonnes : {list(df_mpg.columns)}\n")

        # Affiche les 5 attaquants les mieux notés
        top_fw = (
            df_mpg[df_mpg["position"] == "FW"]
            .dropna(subset=["avg_rating"])
            .nlargest(5, "avg_rating")
        )
        print("🏆 Top 5 attaquants (meilleure note MPG) :")
        print(top_fw[["player_name", "first_name", "team", "price", "avg_rating", "recent_form_avg"]].to_string(index=False))

        # Affiche les 5 joueurs les plus chers
        print("\n💰 Top 5 joueurs les plus chers :")
        top_price = df_mpg.nlargest(5, "price")
        print(top_price[["player_name", "first_name", "position", "price", "avg_rating"]].to_string(index=False))

    else:
        print("\n⚠️ API MPG indisponible. Vérifiez votre connexion internet.")

    # --- Test 2 : FBref CSV ---
    print("\n📊 Test du fichier FBref CSV...")
    df_fbref = load_fbref_csv()

    if not df_fbref.empty:
        print(f"✅ FBref CSV OK : {len(df_fbref)} joueurs")
        print(f"Colonnes disponibles : {list(df_fbref.columns[:10])}...")
    else:
        print("ℹ️  Fichier FBref non disponible (normal si pas encore téléchargé).")
        print("   → Voir les instructions dans la fonction load_fbref_csv()")

    # --- Test 3 : Dataset complet ---
    print("\n🔧 Construction du dataset master...")
    df_master = build_master_dataset()
    print(f"\n📋 Dataset final : {len(df_master)} joueurs, {len(df_master.columns)} colonnes")
    print("Prêt pour l'analyse ! ✨")
