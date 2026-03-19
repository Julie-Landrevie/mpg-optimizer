# ⚽ MPG Optimizer

> Analyse de données et optimisation de stratégie pour **Mon Petit Gazon** — le fantasy football français.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Status](https://img.shields.io/badge/status-en%20développement-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 Objectifs du projet

Ce projet exploite les données disponibles autour de la Ligue 1 pour prendre de meilleures décisions dans MPG :

| Module | Description | Statut |
|--------|-------------|--------|
| 🔌 **Collecte de données** | API MPG officielle + FBref + Understat | 🚧 En cours |
| 📊 **Analyse joueurs** | Stats avancées, forme récente, scoring composite | 🚧 En cours |
| 🧠 **Optimisation compo** | Meilleur XI selon budget & contraintes | 🚧 En cours |
| 💸 **Stratégie mercato** | Détection de pépites & joueurs surcotés | 🚧 En cours |

---

## 🗂️ Structure du projet

```
mpg-optimizer/
├── data/
│   ├── raw/              # Données brutes téléchargées (non versionnées)
│   └── processed/        # Données nettoyées et prêtes à l'analyse
├── notebooks/            # Exploration & prototypage Jupyter
├── src/
│   ├── data/
│   │   └── collect.py        # Collecte : API MPG + FBref + croisement
│   ├── analysis/
│   │   ├── player_rating.py  # Scoring composite des joueurs
│   │   └── mercato.py        # Détection de pépites & analyse de forme
│   └── optimization/
│       └── lineup.py         # Optimisation du XI (programmation linéaire)
├── tests/
│   └── test_player_rating.py # Tests unitaires
├── docs/                 # Documentation technique
├── requirements.txt
└── README.md
```

---

## 📦 Sources de données

Ce projet croise **3 sources complémentaires** pour avoir l'image la plus complète possible :

### 1. 🟢 API officielle MPG — *la source principale*
L'application MPG expose ses propres données via une API publique.
C'est la **source de vérité** : ce sont exactement les cotations et les notes que tu vois dans l'appli.

Ce qu'elle fournit :
- **La cotation** (prix en millions) de chaque joueur
- **La note moyenne** de saison (l'`avg_rating` officielle)
- **L'historique des notes** journée par journée → pour calculer la forme récente
- **Le statut** du joueur (disponible, blessé, suspendu)
- **Le poste** selon le système MPG (GK / DF latéral / DF central / MF déf / MF off / FW)

### 2. 🔵 FBref — *les stats avancées*
[FBref](https://fbref.com) est une base de données football gratuite et très complète.
Elle fournit des statistiques que l'API MPG ne donne pas directement :
- **xG** (expected goals) : qualité des occasions de but créées
- **xA** (expected assists) : qualité des passes menant à des tirs
- Tirs, tirs cadrés, passes progressives, tacles, interceptions...

> 💡 **xG et xA en deux mots :** ces métriques mesurent la *qualité* des actions, pas juste le résultat.
> Un attaquant qui rate 15 occasions faciles a un xG élevé mais peu de vrais buts — c'est utile
> pour savoir s'il va finir par marquer ou si c'est structurel.

### 3. 🔴 Understat *(prévu)*
[Understat](https://understat.com) affine les xG avec des modèles encore plus précis.
Intégration prévue dans une prochaine version.

### Comment les données sont croisées

```
API MPG (cotations + notes)
        │
        ▼
  Normalisation des noms  ────▶  FBref (xG, tirs, passes...)
  (suppression accents,               │
   mise en minuscules)                ▼
        │                    Dataset Master complet
        │                    (une ligne par joueur,
        ▼                     ~30 colonnes)
  Historique notes MPG ───▶  Ajout de la forme récente
  (5 dernières journées)     (recent_form_avg)
```

---

## 🚀 Installation

```bash
# 1. Cloner le repo
git clone https://github.com/<ton-username>/mpg-optimizer.git
cd mpg-optimizer

# 2. Créer un environnement virtuel (bonne pratique pour isoler les dépendances)
python -m venv venv
source venv/bin/activate    # macOS / Linux
# ou : venv\Scripts\activate  # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## 🧪 Utilisation rapide

```python
from src.data.collect import build_master_dataset
from src.analysis.player_rating import compute_ratings, get_top_players
from src.analysis.mercato import compute_value_score, find_undervalued
from src.optimization.lineup import optimize_xi, print_xi

# ── Étape 1 : Récupérer et croiser toutes les données ──
# Cette fonction appelle l'API MPG + FBref et fusionne tout automatiquement
df = build_master_dataset(season=2025)

# ── Étape 2 : Calculer le score composite de chaque joueur ──
# Combine la note MPG officielle + les stats FBref, pondérés par poste
df_rated = compute_ratings(df)

# ── Étape 3 : Voir les meilleurs milieux ──
print(get_top_players(df_rated, position="MF", top_n=10))

# ── Étape 4 : Trouver des pépites (bonne affaire < 20M) ──
df_valued = compute_value_score(df_rated)
pepites = find_undervalued(df_valued, position="MF", max_price=20)
print(pepites)

# ── Étape 5 : Optimiser son XI avec un budget de 500M ──
result = optimize_xi(df_rated, budget=500)
print_xi(result)
```

---

## 📊 Exemple de sortie

```
==================================================
  🏆 XI OPTIMAL — Formation 4-3-3
  Score total : 75.40 | Budget utilisé : 487.0M
==================================================
  [GK] Areola              West Ham        ★ 7.2   10M
  [DF] Saliba              Arsenal         ★ 8.1   30M
  [DF] Truffert            Rennes          ★ 7.4   12M
  ...
```

---

## 🗺️ Roadmap

- [x] Structure initiale du projet
- [x] Collecte via API MPG officielle (cotations, notes, historique)
- [x] Croisement MPG × FBref (stats avancées)
- [x] Module de scoring composite par poste
- [x] Détection de pépites et analyse de forme
- [x] Optimisation du XI par programmation linéaire
- [ ] Tests d'intégration complets
- [ ] Notebook de démonstration Jupyter
- [ ] Interface CLI interactive
- [ ] Gestion des championnats étrangers (Liga, Premier League...)

---

## 🤝 Contribution

Les contributions sont les bienvenues !
Tu as une idée d'amélioration ? Ouvre une **issue** pour en discuter,
ou directement une **pull request** si tu as déjà codé quelque chose.

---

## 📄 Licence

MIT — voir [LICENSE](LICENSE)
