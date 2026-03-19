#!/bin/bash
# ============================================================
# setup_git.sh
# ============================================================
# Ce script initialise ton repo Git local et t'aide à le pousser sur GitHub.
#
# C'est quoi Git ?
#   Git est un système de "versionnage" : il enregistre l'historique
#   de toutes les modifications du code. Tu peux revenir en arrière,
#   voir qui a changé quoi, et travailler à plusieurs sans se marcher dessus.
#
# C'est quoi GitHub ?
#   GitHub est un site qui héberge les repos Git en ligne.
#   C'est là que ton code sera visible par les autres (recruteurs, collègues...).
#
# Comment utiliser ce script ?
#   Dans le terminal, depuis le dossier du projet :
#     bash setup_git.sh <ton-username-github>
#
#   Exemple :
#     bash setup_git.sh marie_dupont
# ============================================================

# $1 = le premier argument passé au script (ton username GitHub)
# Si aucun argument n'est fourni, on utilise "<ton-username>" comme valeur par défaut
GITHUB_USER=${1:-"<ton-username>"}
REPO_NAME="mpg-optimizer"

# "set -e" = le script s'arrête immédiatement si une commande échoue
# (évite de continuer sur une erreur silencieuse)
set -e

echo ""
echo "🚀 Initialisation du repo Git — MPG Optimizer"
echo "=============================================="
echo ""

# ── Étape 1 : Initialiser Git dans le dossier courant ──
# "git init" crée un dossier caché .git/ qui contient tout l'historique
echo "📁 Initialisation du repo Git local..."
git init

# ── Étape 2 : Configurer la branche principale ──
# Par convention moderne, la branche principale s'appelle "main" (et non plus "master")
git checkout -b main 2>/dev/null || git checkout main

# ── Étape 3 : Ajouter tous les fichiers au "staging area" ──
# "git add ." = "je veux inclure tous ces fichiers dans le prochain commit"
# Le .gitignore s'applique ici : les fichiers ignorés ne seront pas ajoutés
echo "📋 Ajout des fichiers au commit..."
git add .

# ── Étape 4 : Vérifier ce qui va être commité ──
echo ""
echo "📄 Fichiers qui seront commités :"
git status --short
echo ""

# ── Étape 5 : Créer le premier commit ──
# Un commit = une "photo" de l'état du code à un instant T, avec un message descriptif
# Convention de message : "type: description courte"
#   feat     = nouvelle fonctionnalité
#   fix      = correction de bug
#   docs     = documentation
#   refactor = restructuration du code sans changement de comportement
#   test     = ajout ou modification de tests
echo "💾 Création du commit initial..."
git commit -m "feat: initial project structure — MPG Optimizer

Sources de données :
- API officielle MPG (cotations, notes, historique journée par journée)
- FBref via soccerdata (xG, xA, stats avancées)
- Croisement automatique par normalisation des noms

Modules :
- src/data/collect.py        : collecte multi-sources + build_master_dataset()
- src/analysis/player_rating.py : scoring composite intégrant avg_rating MPG
- src/analysis/mercato.py    : détection de pépites + analyse de forme récente
- src/optimization/lineup.py : optimisation du XI par programmation linéaire (PuLP)

Tests :
- tests/test_player_rating.py : 20 tests unitaires couvrant cas normaux et edge cases

Documentation :
- README.md avec architecture des données et exemples d'utilisation
- Code entièrement commenté en français pour prise en main facilitée"

echo ""
echo "✅ Commit initial créé avec succès !"
echo ""

# ── Étape 6 : Instructions pour pousser sur GitHub ──
echo "=============================================="
echo "  📡 PROCHAINE ÉTAPE : Pousser sur GitHub"
echo "=============================================="
echo ""
echo "  1️⃣  Va sur https://github.com/new"
echo "  2️⃣  Crée un repo nommé '${REPO_NAME}'"
echo "      ⚠️  NE coche PAS 'Add a README file'"
echo "         (on a déjà le nôtre !)"
echo ""
echo "  3️⃣  Puis reviens ici et exécute ces 3 commandes :"
echo ""
echo "      git remote add origin https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
echo "      git branch -M main"
echo "      git push -u origin main"
echo ""
echo "  🎉 Ton repo sera en ligne sur :"
echo "     https://github.com/${GITHUB_USER}/${REPO_NAME}"
echo ""
echo "=============================================="
echo "  💡 COMMANDES GIT UTILES POUR LA SUITE"
echo "=============================================="
echo ""
echo "  Voir l'état des fichiers modifiés :"
echo "    git status"
echo ""
echo "  Ajouter des modifications et créer un nouveau commit :"
echo "    git add ."
echo "    git commit -m 'feat: description de ce que tu as ajouté'"
echo ""
echo "  Pousser les nouveaux commits sur GitHub :"
echo "    git push"
echo ""
echo "  Voir l'historique des commits :"
echo "    git log --oneline"
echo ""
