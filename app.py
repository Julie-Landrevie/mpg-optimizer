"""
app.py
------
Interface web Streamlit pour MPG Optimizer (français / anglais, clair / sombre).
Charte commune : jl_ui.py et .streamlit/config.toml.
Lancement : streamlit run app.py
"""

import ast
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from jl_ui import footer, fr_num, header, kpi, note, setup, t  # noqa: E402

setup("MPG Optimizer — Julie Landrevie", "MPG Optimizer — Julie Landrevie", icon="⚽")

import pandas as pd  # noqa: E402

from src.analysis.player_rating import compute_ratings, get_value_picks  # noqa: E402
from src.data.collect import build_master_dataset  # noqa: E402


# ============================================================
# CLUBS LIGUE 1 ET LIBELLÉS
# ============================================================

CLUBS = {
    "mpg_championship_club_138":  "Auxerre",
    "mpg_championship_club_141":  "Le Havre",
    "mpg_championship_club_142":  "Lens",
    "mpg_championship_club_143":  "Lyon",
    "mpg_championship_club_144":  "Marseille",
    "mpg_championship_club_145":  "Metz",
    "mpg_championship_club_146":  "Monaco",
    "mpg_championship_club_147":  "Montpellier",
    "mpg_championship_club_149":  "PSG",
    "mpg_championship_club_150":  "Rennes",
    "mpg_championship_club_152":  "Saint-Étienne",
    "mpg_championship_club_153":  "Strasbourg",
    "mpg_championship_club_429":  "Lille",
    "mpg_championship_club_427":  "Toulouse",
    "mpg_championship_club_430":  "Nantes",
    "mpg_championship_club_694":  "Lorient",
    "mpg_championship_club_862":  "Brest",
    "mpg_championship_club_1395": "Nice",
    "mpg_championship_club_1423": "Reims",
    "mpg_championship_club_2128": "Angers",
    "mpg_championship_club_2338": "Paris FC",
}
POS = {"GK": ("Gardiens", "Goalkeepers"), "DF": ("Défenseurs", "Defenders"),
       "MF": ("Milieux", "Midfielders"), "FW": ("Attaquants", "Forwards")}
STATUS_EN = {"Disponible": "Available", "Blessé": "Injured", "Suspendu": "Suspended", "Incertain": "Doubtful"}
ALL = "__all__"


def get_club_name(club_id: str) -> str:
    if club_id in CLUBS:
        return CLUBS[club_id]
    return f"Club #{str(club_id).replace('mpg_championship_club_', '')}"


def status_label(s) -> str:
    return t(str(s), STATUS_EN.get(str(s), str(s)))


def parse_last_ratings(ratings_str) -> list:
    try:
        ratings = ast.literal_eval(str(ratings_str))
        return [r for r in ratings if isinstance(r, (int, float))]
    except Exception:
        return []


def trend(delta: float) -> str:
    if delta > 0.5:
        return t("▲▲ forte hausse", "▲▲ strong rise")
    if delta > 0:
        return t("▲ hausse", "▲ rise")
    if delta > -0.5:
        return t("▬ stable", "▬ stable")
    return t("▼ baisse", "▼ drop")


def price_fmt():
    return t("%d M€", "€%dM")


def show_table(d: pd.DataFrame, cols: dict, progress=None, height=None, fmt2=()):
    """Tableau avec en-têtes traduits ; cols = {colonne: (fr, en)}."""
    keep = [c for c in cols if c in d.columns]
    out = d[keep].copy()
    if "status" in out.columns:
        out["status"] = out["status"].map(status_label)
    out.columns = [t(*cols[c]) for c in keep]
    cfg = {}
    for c in keep:
        name = t(*cols[c])
        if c == "price":
            cfg[name] = st.column_config.NumberColumn(name, format=price_fmt())
        elif c == progress:
            cfg[name] = st.column_config.ProgressColumn(name, min_value=0, max_value=10, format="%.2f")
        elif c in fmt2:
            cfg[name] = st.column_config.NumberColumn(name, format="%.2f")
        elif c == "delta":
            cfg[name] = st.column_config.NumberColumn(name, format="%+.2f")
    kw = dict(hide_index=True, width="stretch", column_config=cfg)
    if height:
        kw["height"] = height
    st.dataframe(out, **kw)


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

@st.cache_data(ttl=3600)
def load_data():
    df = build_master_dataset()
    df_rated = compute_ratings(df, min_minutes=0)
    df_rated["club_name"] = df_rated["team"].apply(get_club_name)
    return df_rated


# ============================================================
# BARRE LATÉRALE
# ============================================================

with st.sidebar:
    st.markdown(f"### {t('Filtres', 'Filters')}")
    selected_position = st.selectbox(t("Poste", "Position"), [ALL] + list(POS),
                                     format_func=lambda k: t("Tous les postes", "All positions") if k == ALL else t(*POS[k]))
    selected_position = None if selected_position == ALL else selected_position
    max_price = st.slider(t("Budget max par joueur (M€)", "Max budget per player (€M)"), 1, 50, 50)
    min_games = st.slider(t("Matchs joués minimum", "Minimum matches played"), 0, 34, 5)
    show_injured = st.checkbox(t("Inclure blessés et suspendus", "Include injured and suspended"), value=False)
    if st.button(t("Actualiser les données", "Refresh data"), width="stretch"):
        st.cache_data.clear()
        st.rerun()
    st.caption(t("Source : API officielle MPG · mise à jour toutes les heures",
                 "Source: official MPG API · refreshed every hour"))


# ============================================================
# CHARGEMENT
# ============================================================

with st.spinner(t("Chargement des données MPG…", "Loading MPG data…")):
    try:
        df = load_data()
    except Exception as e:
        st.error(t(f"Erreur de chargement : {e}", f"Loading error: {e}"))
        st.stop()


# ============================================================
# EN-TÊTE
# ============================================================

header("MPG Optimizer", t("Note et choix des joueurs de Ligue 1 pour Mon Petit Gazon : score composite, bonnes affaires, "
                          "forme et meilleur onze sous contrainte de budget (programmation linéaire).",
                          "Rating and picking Ligue 1 players for the Mon Petit Gazon fantasy game: composite score, "
                          "value picks, form and best eleven under a budget (linear programming)."))

df_scored = df[df["avg_rating"].notna()]
n_total = len(df_scored)
n_dispo = len(df[df["status"] == "Disponible"])
n_blesse = len(df[df["status"] == "Blessé"])

best_row = df_scored.nlargest(1, "mpg_score").iloc[0] if n_total else None
df_form = df_scored[df_scored["recent_form_avg"].notna()].copy()
df_form["delta"] = df_form["recent_form_avg"] - df_form["avg_rating"]
hot_row = df_form.nlargest(1, "delta").iloc[0] if len(df_form) else None

c = st.columns(5)
kpi(c[0], str(n_total), t("Joueurs analysés", "Players analysed"))
kpi(c[1], str(n_dispo), t("Disponibles", "Available"))
kpi(c[2], str(n_blesse), t("Blessés", "Injured"))
kpi(c[3], fr_num(best_row["mpg_score"], 2) if best_row is not None else "—",
    t("Meilleur score", "Best score") + (t(" : ", ": ") + f"{best_row['player_name']}" if best_row is not None else ""), accent=True)
kpi(c[4], ("+" + fr_num(hot_row["delta"], 1)) if hot_row is not None else "—",
    t("Plus en forme", "Best form") + (t(" : ", ": ") + f"{hot_row['player_name']}" if hot_row is not None else ""))
st.write("")


# ============================================================
# ONGLETS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([t("Classement", "Rankings"), t("Bonnes affaires et forme", "Value and form"),
                                        t("Optimiseur de onze", "Eleven optimiser"), t("Recherche", "Search"),
                                        t("Statistiques", "Statistics")])

COLS_MAIN = {"player_name": ("Joueur", "Player"), "first_name": ("Prénom", "First name"), "position": ("Poste", "Pos"),
             "club_name": ("Club", "Club"), "price": ("Prix", "Price"), "avg_rating": ("Note MPG", "MPG rating"),
             "recent_form_avg": ("Forme récente", "Recent form"), "mpg_score": ("Score /10", "Score /10"),
             "games_played": ("Matchs", "Matches"), "started_matches": ("Titulaire", "Starts"),
             "status": ("Statut", "Status")}

# ── ONGLET 1 : Classement ──────────────────────────────────
with tab1:
    st.subheader(t("Classement par score composite", "Ranking by composite score"))
    mask = df["avg_rating"].notna()
    if selected_position:
        mask &= df["position"].str.startswith(selected_position, na=False)
    if not show_injured:
        mask &= df["status"] == "Disponible"
    mask &= df["price"] <= max_price
    mask &= df["games_played"] >= min_games
    df_filt = df[mask].sort_values("mpg_score", ascending=False).reset_index(drop=True)
    st.caption(t(f"{len(df_filt)} joueur(s) affiché(s)", f"{len(df_filt)} player(s) shown"))
    if df_filt.empty:
        st.info(t("Aucun joueur ne correspond aux filtres.", "No player matches the filters."))
    else:
        show_table(df_filt, COLS_MAIN, progress="mpg_score", height=540, fmt2=("avg_rating", "recent_form_avg"))


# ── ONGLET 2 : Bonnes affaires et forme ────────────────────
with tab2:
    col_pep, col_frm = st.columns(2, gap="large")
    with col_pep:
        st.subheader(t("Bonnes affaires : meilleur rapport score / prix", "Value picks: best score-to-price ratio"))
        pos_tabs = st.tabs(list(POS))
        for i, pos in enumerate(POS):
            with pos_tabs[i]:
                pep = get_value_picks(df, position=pos, max_price=max_price, top_n=8)
                if pep.empty:
                    st.info(t("Aucune bonne affaire avec ces filtres.", "No value pick with these filters."))
                else:
                    if "team" in pep.columns:
                        pep["club_name"] = pep["team"].apply(get_club_name)
                    show_table(pep, {"player_name": ("Joueur", "Player"), "club_name": ("Club", "Club"),
                                     "avg_rating": ("Note", "Rating"), "recent_form_avg": ("Forme", "Form"),
                                     "mpg_score": ("Score", "Score"), "price": ("Prix", "Price"),
                                     "value_score": ("Score / prix", "Score / price")},
                               progress="mpg_score", fmt2=("avg_rating", "recent_form_avg", "value_score"))

    with col_frm:
        df_f = df[df["recent_form_avg"].notna() & df["avg_rating"].notna()
                  & (df["games_played"] >= min_games) & (df["status"] == "Disponible")].copy()
        df_f["delta"] = (df_f["recent_form_avg"] - df_f["avg_rating"]).round(2)
        df_f["trend"] = df_f["delta"].apply(trend)
        if selected_position:
            df_f = df_f[df_f["position"].str.startswith(selected_position, na=False)]
        cols_form = {"player_name": ("Joueur", "Player"), "position": ("Poste", "Pos"), "club_name": ("Club", "Club"),
                     "avg_rating": ("Moy.", "Avg"), "recent_form_avg": ("Forme", "Form"), "delta": ("Écart", "Change"),
                     "trend": ("Tendance", "Trend"), "price": ("Prix", "Price")}
        st.subheader(t("En montée de forme", "Rising form"))
        show_table(df_f.nlargest(8, "delta"), cols_form, fmt2=("avg_rating", "recent_form_avg"))
        st.subheader(t("En baisse", "Dropping"))
        show_table(df_f.nsmallest(6, "delta"), cols_form, fmt2=("avg_rating", "recent_form_avg"))
        st.caption(t("Forme : moyenne des 5 dernières notes ; écart : forme moins moyenne de la saison.",
                     "Form: average of the last 5 ratings; change: form minus season average."))


# ── ONGLET 3 : Optimiseur de onze ──────────────────────────
with tab3:
    st.subheader(t("Construis ton meilleur onze", "Build your best eleven"))
    st.caption(t("L’optimiseur choisit les 11 joueurs qui maximisent le score total selon ton budget et ta formation.",
                 "The optimiser picks the 11 players who maximise the total score for your budget and formation."))
    col_params1, col_params2, col_params3 = st.columns(3)
    with col_params1:
        budget_xi = st.slider(t("Budget total (M€)", "Total budget (€M)"), min_value=100, max_value=500, value=500, step=10)
    with col_params2:
        formation_options = {"auto": None, "4-4-2": (4, 4, 2), "4-3-3": (4, 3, 3), "4-5-1": (4, 5, 1), "3-5-2": (3, 5, 2),
                             "3-4-3": (3, 4, 3), "5-3-2": (5, 3, 2), "5-4-1": (5, 4, 1)}
        formation_key = st.selectbox(t("Formation", "Formation"), list(formation_options),
                                     format_func=lambda k: t("Automatique (la meilleure)", "Automatic (best)") if k == "auto" else k)
        selected_formation = formation_options[formation_key]
    with col_params3:
        only_available = st.checkbox(t("Joueurs disponibles uniquement", "Available players only"), value=True)
        min_games_xi = st.slider(t("Matchs joués minimum", "Minimum matches played"), 0, 20, 5, key="min_games_xi")

    with st.expander(t("Options avancées : joueurs imposés ou exclus", "Advanced options: locked or excluded players")):
        col_lock, col_excl = st.columns(2)
        with col_lock:
            locked_input = st.text_input(t("Joueurs obligatoires (séparés par des virgules)", "Locked players (comma-separated)"),
                                         key="locked", placeholder=t("Ex. : Tolisso, Pagis", "E.g. Tolisso, Pagis"))
        with col_excl:
            excluded_input = st.text_input(t("Joueurs à exclure (séparés par des virgules)", "Excluded players (comma-separated)"),
                                           key="excluded", placeholder=t("Ex. : Greenwood, Dembélé", "E.g. Greenwood, Dembélé"))
    locked_players = [n.strip() for n in locked_input.split(",") if n.strip()]
    excluded_players = [n.strip() for n in excluded_input.split(",") if n.strip()]

    if st.button(t("Générer le meilleur onze", "Generate the best eleven"), type="primary", width="stretch"):
        df_xi = df[df["avg_rating"].notna() & (df["games_played"] >= min_games_xi)].copy()
        if only_available:
            df_xi = df_xi[df_xi["status"] == "Disponible"]
        if excluded_players:
            df_xi = df_xi[~df_xi["player_name"].isin(excluded_players)]

        counts = df_xi.groupby("position").size()
        ok = True
        for pos, needed in [("GK", 1), ("DF", 3), ("MF", 3), ("FW", 1)]:
            available = sum(counts.get(p, 0) for p in counts.index if p.startswith(pos))
            if available < needed:
                st.error(t(f"Pas assez de {pos} disponibles ({available} < {needed}). Élargis les filtres.",
                           f"Not enough {pos} available ({available} < {needed}). Widen the filters."))
                ok = False

        if ok:
            try:
                from src.optimization.lineup import optimize_xi
                with st.spinner(t("Calcul du onze optimal…", "Computing the optimal eleven…")):
                    result = optimize_xi(df_xi, budget=budget_xi, formation=selected_formation,
                                         locked_players=locked_players, excluded_players=[])
                if not result:
                    st.error(t("Aucune solution trouvée. Augmente le budget ou élargis les filtres.",
                               "No solution found. Increase the budget or widen the filters."))
                else:
                    r = st.columns(3)
                    kpi(r[0], result["formation"], t("Formation retenue", "Chosen formation"))
                    kpi(r[1], fr_num(result["total_score"], 2), t("Score total", "Total score"), accent=True)
                    kpi(r[2], f"{result['total_price']:.0f} / {budget_xi}", t("Budget utilisé (M€)", "Budget used (€M)"))
                    st.write("")

                    players_df = result["players"].copy()
                    players_df["club_name"] = players_df["team"].apply(get_club_name)

                    def display_line(rows):
                        cols = st.columns(max(1, len(rows)))
                        for i, (_, pl) in enumerate(rows.iterrows()):
                            prenom = pl.get("first_name", "")
                            nom = str(pl.get("player_name", "") or "")
                            nom_affiche = f"{str(prenom)[:1]}. {nom}" if pd.notna(prenom) and str(prenom).strip() else nom
                            cols[i].markdown(
                                f'<div class="jl-kpi" style="text-align:center;padding:0.6rem 0.3rem">'
                                f'<div style="font-weight:700;color:var(--jl-ink)">{nom_affiche}</div>'
                                f'<div class="l">{pl["club_name"]}</div>'
                                f'<div style="color:var(--jl-teal);font-weight:700">{fr_num(pl["mpg_score"], 1)}</div>'
                                f'<div class="l">{pl["price"]} M€</div></div>', unsafe_allow_html=True)

                    st.subheader(t("Composition", "Line-up"))
                    for code in ["FW", "MF", "DF", "GK"]:
                        rows = players_df[players_df["position"].str.startswith(code)]
                        st.markdown(f"**{t(*POS[code])}**")
                        display_line(rows)

                    st.subheader(t("Détail du onze", "Eleven detail"))
                    show_table(players_df, {k: COLS_MAIN[k] for k in ["player_name", "first_name", "position", "club_name",
                                                                       "avg_rating", "recent_form_avg", "mpg_score",
                                                                       "price", "status"]},
                               progress="mpg_score", fmt2=("avg_rating", "recent_form_avg"))
                    note(t(f"Budget restant : <b>{budget_xi - result['total_price']:.0f} M€</b> sur {budget_xi} M€.",
                           f"Remaining budget: <b>€{budget_xi - result['total_price']:.0f}M</b> out of €{budget_xi}M."))
            except ImportError:
                st.error(t("Module d’optimisation indisponible. Vérifie que PuLP est installé : pip install pulp",
                           "Optimisation module unavailable. Check that PuLP is installed: pip install pulp"))
            except Exception as e:
                st.error(t(f"Erreur : {e}", f"Error: {e}"))
    else:
        st.info(t("Règle les paramètres puis clique sur « Générer le meilleur onze ».",
                  "Set the parameters, then click “Generate the best eleven”."))


# ── ONGLET 4 : Recherche ───────────────────────────────────
with tab4:
    st.subheader(t("Rechercher un joueur", "Search for a player"))
    search = st.text_input(t("Nom", "Name"), placeholder=t("Ex. : Tolisso, Pagis, Greenwood…", "E.g. Tolisso, Pagis, Greenwood…"),
                           label_visibility="collapsed")
    if search and len(search) >= 2:
        mask_s = (df["player_name"].str.contains(search, case=False, na=False)
                  | df["first_name"].str.contains(search, case=False, na=False))
        results = df[mask_s]
        if results.empty:
            st.info(t(f"Aucun joueur trouvé pour « {search} ».", f"No player found for “{search}”."))
        else:
            for _, row in results.iterrows():
                prenom = str(row.get("first_name", "") or "")
                titre = f"{prenom} {row['player_name']} · {row['position']} · {get_club_name(row['team'])} · {row['price']} M€"
                with st.expander(titre):
                    c = st.columns(4)

                    def v2(col):
                        return fr_num(row[col], 2) if pd.notna(row.get(col)) else "—"
                    kpi(c[0], v2("avg_rating"), t("Note MPG", "MPG rating"))
                    kpi(c[1], v2("recent_form_avg"), t("Forme récente", "Recent form"))
                    kpi(c[2], v2("mpg_score"), t("Score composite", "Composite score"))
                    kpi(c[3], status_label(row.get("status", "—")), t("Statut", "Status"))
                    st.write("")
                    ca, cb = st.columns(2)
                    with ca:
                        st.markdown(f"{t('Matchs joués', 'Matches played')}{t(' :', ':')} **{int(row.get('games_played', 0))}**  \n"
                                    f"{t('Titularisations', 'Starts')}{t(' :', ':')} **{int(row.get('started_matches', 0))}**  \n"
                                    f"{t('Buts', 'Goals')}{t(' :', ':')} **{int(row.get('total_goals', 0))}**")
                    with cb:
                        st.markdown(f"{t('Clean sheets', 'Clean sheets')}{t(' :', ':')} **{int(row.get('clean_sheets', 0))}**  \n"
                                    f"{t('Cartons jaunes', 'Yellow cards')}{t(' :', ':')} **{int(row.get('yellow_cards', 0))}**  \n"
                                    f"{t('Cartons rouges', 'Red cards')}{t(' :', ':')} **{int(row.get('red_cards', 0))}**")
                    ratings = parse_last_ratings(row.get("last_5_ratings", "[]"))
                    if ratings:
                        st.markdown(f"**{t('5 dernières notes', 'Last 5 ratings')}**")
                        rcols = st.columns(len(ratings))
                        for j, rr in enumerate(ratings):
                            kpi(rcols[j], fr_num(rr, 1), t(f"J-{len(ratings) - j}", f"MD-{len(ratings) - j}"))
    else:
        st.info(t("Tape au moins 2 lettres pour rechercher.", "Type at least 2 letters to search."))


# ── ONGLET 5 : Statistiques ────────────────────────────────
with tab5:
    st.subheader(t("Vue d’ensemble de la Ligue 1", "Ligue 1 overview"))
    df_s = df[df["avg_rating"].notna() & (df["games_played"] >= 5)].copy()
    col_a, col_b = st.columns(2, gap="large")
    with col_a:
        st.markdown(f"**{t('Note moyenne par poste', 'Average rating by position')}**")
        avg_pos = df_s.groupby("position")["avg_rating"].mean().round(2)
        avg_pos.index.name = t("Poste", "Position")
        st.bar_chart(avg_pos.rename(t("Note moyenne", "Average rating")))
        st.markdown(f"**{t('Meilleurs buteurs', 'Top scorers')}**")
        show_table(df_s[df_s["total_goals"] > 0].nlargest(10, "total_goals"),
                   {"player_name": ("Joueur", "Player"), "club_name": ("Club", "Club"), "position": ("Poste", "Pos"),
                    "total_goals": ("Buts", "Goals"), "price": ("Prix", "Price")})
    with col_b:
        st.markdown(f"**{t('Répartition des cotes', 'Price distribution')}**")
        price_dist = df[df["avg_rating"].notna()]["price"].value_counts().sort_index()
        price_dist.index.name = t("Cote (M€)", "Price (€M)")
        st.bar_chart(price_dist.rename(t("Joueurs", "Players")))
        st.markdown(f"**{t('Gardiens : clean sheets', 'Goalkeepers: clean sheets')}**")
        show_table(df_s[df_s["position"] == "GK"].nlargest(8, "clean_sheets"),
                   {"player_name": ("Gardien", "Goalkeeper"), "club_name": ("Club", "Club"),
                    "clean_sheets": ("Clean sheets", "Clean sheets"), "avg_rating": ("Note", "Rating"),
                    "price": ("Prix", "Price")}, fmt2=("avg_rating",))

footer("Données : API MPG (Mon Petit Gazon)", "Data: MPG API (Mon Petit Gazon)")
