# ⚽ MPG Optimizer

> Analyse de données et optimisation de stratégie pour **Mon Petit Gazon** — le fantasy football français.

🌐 **[Ouvrir l'application](https://mpg-optimizer.streamlit.app)** — mpg-optimizer.streamlit.app

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![Status](https://img.shields.io/badge/status-en_production-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🌐 Application en ligne

**👉 [mpg-optimizer.streamlit.app](https://mpg-optimizer.streamlit.app)**

L'application est accessible publiquement — aucune installation requise.
Elle se met à jour automatiquement depuis l'API MPG officielle toutes les heures.

---

## 🎯 Objectifs du projet

Ce projet exploite les données disponibles autour de la Ligue 1 pour prendre de meilleures décisions dans MPG :

| Module | Description | Statut |
|--------|-------------|--------|
| 🔌 **Collecte de données** | API MPG officielle + FBref (520 joueurs croisés) | ✅ Fait |
| 📊 **Scoring joueurs** | Notes MPG + forme récente + stats, pondérés par poste | ✅ Fait |
| 🧠 **Optimiseur XI** | Meilleur XI selon budget, formation & contraintes | ✅ Fait |
| 💸 **Pépites & forme** | Détection des bonnes affaires et joueurs en montée | ✅ Fait |
| 📓 **Analyse exploratoire** | Notebook Jupyter avec graphiques et insights | ✅ Fait |

---

## 🗂️ Structure du projet

```
mpg-optimizer/
├── app.py                    # Interface Streamlit (point d'entrée)
├── requirements.txt
├── data/
│   └── raw/                  # Données brutes (non versionnées)
├── notebooks/
│   └── mpg_analyse_exploratoire.ipynb  # Analyse exploratoire complète
├── src/
│   ├── data/
│   │   └── collect.py        # Collecte : API MPG + FBref + croisement 3 passes
│   ├── analysis/
│   │   ├── player_rating.py  # Scoring composite des joueurs
│   │   └── mercato.py        # Détection de pépites & analyse de forme
│   └── optimization/
│       └── lineup.py         # Optimisation du XI (programmation linéaire)
└── tests/
    └── test_player_rating.py
```

---

## 📦 Sources de données

Ce projet croise **2 sources complémentaires** :

### 🟢 API officielle MPG — *la source principale*
L'application MPG expose ses propres données via une API publique.
C'est la **source de vérité** : cotations et notes officielles de l'appli.

- Cotation (prix en millions) de chaque joueur
- Note moyenne de saison (`avg_rating`)
- Historique des 5 dernières notes → forme récente
- Statut (disponible, blessé, suspendu)
- Poste selon le système MPG (GK / DF / MF / FW)

### 🔵 FBref — *les stats avancées*
[FBref](https://fbref.com) fournit des statistiques complémentaires :
buts, passes décisives, minutes jouées, et métriques per 90 minutes.

**Matching MPG × FBref : algorithme 3 passes**
```
Passe 1 : prénom + nom exact     → 417 joueurs
Passe 2 : nom seul unique        →  67 joueurs
Passe 3 : lookup dictionnaire    →  36 joueurs
─────────────────────────────────────────────
Total : 520 joueurs croisés sur 514 (était 27 avant)
```

---

## 🚀 Installation locale

```bash
# 1. Cloner le repo
git clone https://github.com/Julie-Landrevie/mpg-optimizer.git
cd mpg-optimizer

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate    # macOS / Linux

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'application
streamlit run app.py
```

---

## 🧪 Utilisation en ligne de commande

```bash
# Collecter les données MPG + FBref
python -m src.data.collect

# Calculer les scores et afficher les tops
python -m src.analysis.player_rating
```

---

## 📊 Insights clés (notebook exploratoire)

- **G+A** (buts + passes) est la stat FBref la plus corrélée avec la note MPG (r = 0.49)
- **Beraldo (PSG)** : 6M€ pour un score de 8.32 → meilleure affaire défensive
- **Pagis (Lorient)** : meilleur attaquant de la saison (score 8.53, 21M€)
- **Paris FC** domine la forme récente en fin de saison

---

## 🗺️ Roadmap

- [x] Collecte via API MPG officielle
- [x] Croisement MPG × FBref (520 joueurs)
- [x] Scoring composite par poste
- [x] Détection de pépites et analyse de forme
- [x] Optimiseur XI avec contraintes budget/formation
- [x] Interface Streamlit déployée en ligne
- [x] Notebook d'analyse exploratoire
- [ ] Amélioration des poids de scoring via calibration
- [ ] Gestion des championnats étrangers (Liga, Premier League...)
- [ ] Prédiction de note pour la prochaine journée (ML)

---

## 📄 Licence

MIT — voir [LICENSE](LICENSE)
