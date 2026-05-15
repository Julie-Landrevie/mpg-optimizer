"""
app.py
------
Interface web Streamlit pour MPG Optimizer.
Lancement : streamlit run app.py
"""

import streamlit as st
import pandas as pd
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.collect import build_master_dataset
from src.analysis.player_rating import compute_ratings, get_value_picks

# ============================================================
# CONFIGURATION PAGE
# ============================================================

st.set_page_config(
    page_title="MPG Optimizer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    line-height: 1;
}
.hero-sub {
    font-size: 0.9rem;
    color: #94a3b8;
    margin-top: 6px;
}

div[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 12px 16px;
}
div[data-testid="metric-container"] label {
    font-size: 0.72rem !important;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f172a;
    border-left: 4px solid #3b82f6;
    padding-left: 10px;
    margin: 20px 0 10px 0;
}

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
.stDeployButton { display: none; }

section[data-testid="stSidebar"] { background: #0f172a; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CLUBS LIGUE 1 — dictionnaire complet
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

def get_club_name(club_id: str) -> str:
    if club_id in CLUBS:
        return CLUBS[club_id]
    num = club_id.replace("mpg_championship_club_", "")
    return f"Club #{num}"

def parse_last_ratings(ratings_str) -> list:
    try:
        ratings = ast.literal_eval(str(ratings_str))
        return [r for r in ratings if isinstance(r, (int, float))]
    except:
        return []

def form_emoji(delta: float) -> str:
    if delta > 0.5:  return "🔥"
    if delta > 0:    return "📈"
    if delta > -0.5: return "➡️"
    return "❄️"


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
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Filtres")
    st.divider()

    position_options = {
        "Toutes positions": None,
        "🥅 Gardiens":    "GK",
        "🛡️ Défenseurs": "DF",
        "⚙️ Milieux":    "MF",
        "⚡ Attaquants":  "FW",
    }
    position_label    = st.selectbox("Poste", list(position_options.keys()))
    selected_position = position_options[position_label]

    max_price    = st.slider("Budget max / joueur (M€)", 1, 50, 50)
    min_games    = st.slider("Matchs joués minimum", 0, 34, 5)
    show_injured = st.checkbox("Inclure blessés / suspendus", value=False)

    st.divider()

    if st.button("🔄 Actualiser les données", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("""
    <div style="font-size:0.72rem; color:#475569; margin-top:12px; line-height:1.6;">
    Source : API officielle MPG<br>
    Mise à jour toutes les heures
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# CHARGEMENT
# ============================================================

with st.spinner("Chargement des données MPG..."):
    try:
        df = load_data()
    except Exception as e:
        st.error(f"❌ Erreur de chargement : {e}")
        st.stop()


# ============================================================
# EN-TÊTE
# ============================================================

st.markdown("""
<div class="hero">
    <div style="font-size:3rem">⚽</div>
    <div>
        <div class="hero-title">MPG Optimizer</div>
        <div class="hero-sub">Analyse & scoring des joueurs de Ligue 1 · Données temps réel</div>
    </div>
</div>
""", unsafe_allow_html=True)

df_scored = df[df["avg_rating"].notna()]
n_total   = len(df_scored)
n_dispo   = len(df_scored[df_scored["status"] == "Disponible"])
n_blesse  = len(df_scored[df_scored["status"] == "Blessé"])

best_row   = df_scored.nlargest(1, "mpg_score").iloc[0] if n_total else None
best_label = f"{best_row['player_name']} ({best_row['mpg_score']:.2f})" if best_row is not None else "—"

df_form = df_scored[df_scored["recent_form_avg"].notna()].copy()
df_form["delta"] = df_form["recent_form_avg"] - df_form["avg_rating"]
hot_row   = df_form.nlargest(1, "delta").iloc[0] if len(df_form) else None
hot_label = f"{hot_row['player_name']} (+{hot_row['delta']:.1f})" if hot_row is not None else "—"

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Joueurs analysés", n_total)
c2.metric("Disponibles",      n_dispo)
c3.metric("Blessés",          n_blesse)
c4.metric("🏆 Meilleur score", best_label)
c5.metric("🔥 En forme",       hot_label)

st.divider()


# ============================================================
# ONGLETS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Classement",
    "🔥 Pépites & Forme",
    "🔍 Recherche",
    "📈 Statistiques",
])


# ── ONGLET 1 : Classement ──────────────────────────────────
with tab1:
    st.markdown('<div class="section-title">Classement par score composite</div>', unsafe_allow_html=True)

    mask = df["avg_rating"].notna()
    if selected_position:
        mask &= df["position"].str.startswith(selected_position, na=False)
    if not show_injured:
        mask &= df["status"] == "Disponible"
    mask &= df["price"] <= max_price
    mask &= df["games_played"] >= min_games

    df_filt = df[mask].sort_values("mpg_score", ascending=False).reset_index(drop=True)
    st.caption(f"{len(df_filt)} joueur(s) affiché(s)")

    if df_filt.empty:
        st.info("Aucun joueur ne correspond aux filtres.")
    else:
        rename_map = {
            "player_name":     "Joueur",
            "first_name":      "Prénom",
            "position":        "Poste",
            "club_name":       "Club",
            "price":           "Prix (M€)",
            "avg_rating":      "Note MPG",
            "recent_form_avg": "Forme récente",
            "mpg_score":       "Score /10",
            "games_played":    "Matchs",
            "started_matches": "Titulaire",
            "status":          "Statut",
        }
        cols_show  = [c for c in rename_map if c in df_filt.columns]
        df_display = df_filt[cols_show].rename(columns=rename_map)

        for col in ["Note MPG", "Forme récente", "Score /10"]:
            if col in df_display.columns:
                df_display[col] = df_display[col].round(2)

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            height=540,
            column_config={
                "Score /10": st.column_config.ProgressColumn(
                    "Score /10", min_value=0, max_value=10, format="%.2f"
                ),
                "Note MPG":      st.column_config.NumberColumn(format="%.2f"),
                "Forme récente": st.column_config.NumberColumn(format="%.2f"),
                "Prix (M€)":     st.column_config.NumberColumn(format="%d M€"),
            },
        )


# ── ONGLET 2 : Pépites & Forme ─────────────────────────────
with tab2:
    col_pep, col_frm = st.columns([1, 1], gap="large")

    with col_pep:
        st.markdown('<div class="section-title">🔥 Pépites — meilleur rapport score/prix</div>', unsafe_allow_html=True)
        pos_tabs = st.tabs(["🥅 GK", "🛡️ DF", "⚙️ MF", "⚡ FW"])

        for i, pos in enumerate(["GK", "DF", "MF", "FW"]):
            with pos_tabs[i]:
                pep = get_value_picks(df, position=pos, max_price=max_price, top_n=8)
                if pep.empty:
                    st.info("Aucune pépite avec ces filtres.")
                else:
                    if "team" in pep.columns:
                        pep["Club"] = pep["team"].apply(get_club_name)
                    cols_p = ["player_name", "Club", "avg_rating", "recent_form_avg",
                              "mpg_score", "price", "value_score"]
                    cols_p = [c for c in cols_p if c in pep.columns]
                    pep_d = pep[cols_p].rename(columns={
                        "player_name": "Joueur", "avg_rating": "Note",
                        "recent_form_avg": "Forme", "mpg_score": "Score",
                        "price": "Prix", "value_score": "Valeur/Prix",
                    })
                    for c in ["Note", "Forme", "Score", "Valeur/Prix"]:
                        if c in pep_d.columns:
                            pep_d[c] = pep_d[c].round(2)
                    st.dataframe(
                        pep_d, use_container_width=True, hide_index=True,
                        column_config={
                            "Score": st.column_config.ProgressColumn(
                                "Score", min_value=0, max_value=10, format="%.2f"
                            ),
                            "Prix": st.column_config.NumberColumn(format="%d M€"),
                        }
                    )

    with col_frm:
        st.markdown('<div class="section-title">📈 Joueurs en montée de forme</div>', unsafe_allow_html=True)

        df_f = df[
            df["recent_form_avg"].notna() &
            df["avg_rating"].notna() &
            (df["games_played"] >= min_games) &
            (df["status"] == "Disponible")
        ].copy()
        df_f["delta"]    = (df_f["recent_form_avg"] - df_f["avg_rating"]).round(2)
        df_f["Tendance"] = df_f["delta"].apply(form_emoji)

        if selected_position:
            df_f = df_f[df_f["position"].str.startswith(selected_position, na=False)]

        top_f = df_f.nlargest(8, "delta")[
            ["player_name", "position", "club_name", "avg_rating",
             "recent_form_avg", "delta", "Tendance", "price"]
        ].rename(columns={
            "player_name": "Joueur", "position": "Poste", "club_name": "Club",
            "avg_rating": "Moy.", "recent_form_avg": "Forme",
            "delta": "Écart", "price": "Prix",
        })
        st.dataframe(
            top_f, use_container_width=True, hide_index=True,
            column_config={
                "Écart": st.column_config.NumberColumn(format="%+.2f"),
                "Prix":  st.column_config.NumberColumn(format="%d M€"),
            }
        )

        st.markdown('<div class="section-title">❄️ Joueurs en baisse</div>', unsafe_allow_html=True)
        bot_f = df_f.nsmallest(6, "delta")[
            ["player_name", "position", "club_name", "avg_rating",
             "recent_form_avg", "delta", "Tendance", "price"]
        ].rename(columns={
            "player_name": "Joueur", "position": "Poste", "club_name": "Club",
            "avg_rating": "Moy.", "recent_form_avg": "Forme",
            "delta": "Écart", "price": "Prix",
        })
        st.dataframe(
            bot_f, use_container_width=True, hide_index=True,
            column_config={
                "Écart": st.column_config.NumberColumn(format="%+.2f"),
                "Prix":  st.column_config.NumberColumn(format="%d M€"),
            }
        )


# ── ONGLET 3 : Recherche ───────────────────────────────────
with tab3:
    st.markdown('<div class="section-title">🔍 Rechercher un joueur</div>', unsafe_allow_html=True)
    search = st.text_input("", placeholder="Ex: Tolisso, Pagis, Greenwood...")

    if search and len(search) >= 2:
        mask_s = (
            df["player_name"].str.contains(search, case=False, na=False) |
            df["first_name"].str.contains(search, case=False, na=False)
        )
        results = df[mask_s]

        if results.empty:
            st.info(f"Aucun joueur trouvé pour « {search} ».")
        else:
            for _, row in results.iterrows():
                club   = get_club_name(row["team"])
                prenom = str(row.get("first_name", "") or "")
                titre  = f"{prenom} {row['player_name']} — {row['position']} | {club} | {row['price']}M€"

                with st.expander(titre):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Note MPG",        f"{row['avg_rating']:.2f}"      if pd.notna(row.get('avg_rating'))      else "—")
                    c2.metric("Forme récente",   f"{row['recent_form_avg']:.2f}" if pd.notna(row.get('recent_form_avg')) else "—")
                    c3.metric("Score composite", f"{row['mpg_score']:.2f}"       if pd.notna(row.get('mpg_score'))       else "—")
                    c4.metric("Statut",          row.get("status", "—"))

                    ca, cb = st.columns(2)
                    with ca:
                        st.write(f"**Matchs joués :** {int(row.get('games_played', 0))}")
                        st.write(f"**Matchs titulaire :** {int(row.get('started_matches', 0))}")
                        st.write(f"**Buts :** {int(row.get('total_goals', 0))}")
                    with cb:
                        st.write(f"**Clean sheets :** {int(row.get('clean_sheets', 0))}")
                        st.write(f"**Cartons jaunes :** {int(row.get('yellow_cards', 0))}")
                        st.write(f"**Cartons rouges :** {int(row.get('red_cards', 0))}")

                    ratings = parse_last_ratings(row.get("last_5_ratings", "[]"))
                    if ratings:
                        st.write("**5 dernières notes :**")
                        rcols = st.columns(len(ratings))
                        for j, r in enumerate(ratings):
                            emoji = "🟢" if r >= 6 else "🟡" if r >= 5 else "🔴"
                            rcols[j].metric(f"J-{len(ratings)-j}", f"{emoji} {r}")
    else:
        st.info("Tape au moins 2 lettres pour rechercher.")


# ── ONGLET 4 : Statistiques ────────────────────────────────
with tab4:
    st.markdown('<div class="section-title">Vue d\'ensemble de la Ligue 1</div>', unsafe_allow_html=True)

    df_s = df[df["avg_rating"].notna() & (df["games_played"] >= 5)].copy()

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.subheader("Note moyenne par poste")
        avg_pos = (
            df_s.groupby("position")["avg_rating"]
            .mean().round(2).reset_index()
            .rename(columns={"position": "Poste", "avg_rating": "Note moyenne"})
            .set_index("Poste")
        )
        st.bar_chart(avg_pos)

        st.subheader("Top 10 buteurs")
        top_but = (
            df_s[df_s["total_goals"] > 0]
            .nlargest(10, "total_goals")
            [["player_name", "club_name", "position", "total_goals", "price"]]
            .rename(columns={
                "player_name": "Joueur", "club_name": "Club",
                "position": "Poste", "total_goals": "Buts", "price": "Prix",
            })
        )
        st.dataframe(top_but, use_container_width=True, hide_index=True,
                     column_config={"Prix": st.column_config.NumberColumn(format="%d M€")})

    with col_b:
        st.subheader("Distribution des cotations")
        price_dist = df[df["avg_rating"].notna()]["price"].value_counts().sort_index()
        st.bar_chart(price_dist)

        st.subheader("Top gardiens — Clean sheets")
        top_gk = (
            df_s[df_s["position"] == "GK"]
            .nlargest(8, "clean_sheets")
            [["player_name", "club_name", "clean_sheets", "avg_rating", "price"]]
            .rename(columns={
                "player_name": "Gardien", "club_name": "Club",
                "clean_sheets": "CS", "avg_rating": "Note", "price": "Prix",
            })
        )
        st.dataframe(top_gk, use_container_width=True, hide_index=True,
                     column_config={
                         "Note": st.column_config.NumberColumn(format="%.2f"),
                         "Prix": st.column_config.NumberColumn(format="%d M€"),
                     })
