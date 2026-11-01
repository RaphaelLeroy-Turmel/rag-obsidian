---
tags: [projet/rag-obsidian, readme]
created: 2026-11-01
updated: 2026-11-01
statut: en cours — étape 1 (nettoyage des notes)
---

# RAG Obsidian

Un RAG (Retrieval-Augmented Generation) léger qui indexe un dossier de mes notes Obsidian et répond à des questions en citant les notes sources.

> [!info] Fichiers liés
> - [[HISTORIQUE]] : évolution du projet, choix et erreurs, dans l'ordre chronologique
> - [[GIT]] : commandes Git vues pendant le projet et prochaines à apprendre

## Objectif

- Interroger en langage naturel un dossier dédié de notes (`.md`, puis `.pdf`, en français et en anglais).
- Indexer automatiquement les nouveaux fichiers.
- Cible finale : tourner sur mon homelab (moins de 4 Go de RAM disponibles). **Pour l'instant, tout le développement se fait sur mon PC.**

## Architecture visée

```
Dossier de notes (.md, .pdf)
        │
        ▼
  1. Nettoyage        ← étape en cours
  2. Chunking (découpage par titres)
  3. Embeddings (multilingual-e5-small, CPU)
  4. Stockage vectoriel (LanceDB)
  5. Indexation incrémentale (hash + registre SQLite)
        │
        ▼
  Question → recherche des passages → LLM via API → réponse + sources
```

Les raisons de ces choix sont détaillées dans [[HISTORIQUE#2026-10-29 — Choix de l'architecture]].

## État d'avancement

- [x] Environnement Python et dépôt GitHub
- [x] Première version du nettoyage des `.md` (`cleaning.py`)
- [ ] Nettoyage fiabilisé et testé (pytest)
- [ ] Chunking
- [ ] Embeddings
- [ ] Recherche en mémoire (première baseline)
- [ ] Stockage LanceDB + indexation incrémentale
- [ ] Extraction des PDF
- [ ] Génération avec un LLM
- [ ] API (FastAPI) et évaluation (recall@k, MRR)
- [ ] Déploiement sur le homelab

## Structure actuelle du dépôt

```
rag-obsidian/
├── cleaning.py         # nettoyage des notes .md
├── requirements.txt    # dépendances avec versions figées
├── .env.example        # variables attendues, sans valeurs
├── .env                # valeurs locales (ignoré par Git)
├── .gitignore
├── .gitattributes      # fins de ligne normalisées en LF
├── README.md
└── data/               # notes de test (ignoré par Git)
```

## Installation

Prérequis : Anaconda (Windows), Git.

```
conda create -n rag-obsidian python=3.12
conda activate rag-obsidian
python -m pip install -r requirements.txt
```

Puis copier `.env.example` en `.env` et remplir les valeurs.

> [!warning] Règles de l'environnement
> - Installer les paquets avec `python -m pip`, jamais avec `conda install` dans cet environnement.
> - Lancer les scripts avec `python`, jamais avec `py` (qui ignore l'environnement conda).
> - Dans VS Code, sélectionner l'interpréteur `envs\rag-obsidian\python.exe`.

### Variables d'environnement

| Variable | Rôle |
|---|---|
| `STORAGE_DIR` | Dossier contenant les notes à nettoyer |
| `CLEAN_DIR` | Dossier où écrire les notes nettoyées (débogage) |

### Dépendances principales

| Paquet | Usage |
|---|---|
| `python-frontmatter` | Séparer le YAML (propriétés Obsidian) du corps de la note |
| `python-dotenv` | Lire le fichier `.env` (installé, pas encore utilisé dans le code) |
| `pytest` | Tests (installé, pas encore de tests dans le dépôt) |

## Utilisation

```
python cleaning.py
```

Le script charge le `.env`, parcourt `STORAGE_DIR`, nettoie chaque `.md` et affiche un aperçu de chaque note nettoyée.

## Fonctionnement de `cleaning.py`

| Élément | Rôle |
|---|---|
| `cleanFile` (dataclass) | Une note nettoyée : `name`, `content`, `metadata` |
| `load_env_file()` | Lit le `.env` à la main et remplit `os.environ` |
| `clean_wikilinks()` | Remplace `[[cible]]`, `[[cible#section]]`, `[[cible\|alias]]` et `![[embed]]` par le nom de la cible |
| `clean_files()` | Parcourt le dossier, nettoie les wikilinks, sépare le frontmatter, construit les `cleanFile` |
| `main()` | Point d'entrée, protégé par `if __name__ == "__main__"` |

## Limites et bugs connus

Relevés lors de la revue du 2026-11-01 (détail dans [[HISTORIQUE#2026-11-01 — Revue du code de nettoyage]]).

**Bugs**
- Le fichier nettoyé est écrit puis immédiatement supprimé (`write_text` suivi de `unlink`) : rien n'est conservé dans `CLEAN_DIR`.
- Le chemin de sortie colle le nom du fichier au dossier sans séparateur (`f"{clean_dir_path}{file.name}_cleaned"`).
- `main()` affiche les **valeurs** du `.env` : une future clé API apparaîtrait dans le terminal.
- Une ligne sans `=` dans le `.env` fait planter `load_env_file` (`ValueError`).
- Si `STORAGE_DIR` est absent, l'erreur est un `TypeError` peu explicite.
- Le message d'erreur du `.env` contient un `$` en trop (`${filepath}`).

**Comportements à décider ou corriger**
- Les wikilinks sont nettoyés **avant** la lecture du frontmatter : un lien dans les propriétés YAML est aussi modifié.
- Les embeds `![[image.png]]` laissent le nom du fichier dans le texte (le commentaire d'en-tête prévoit de les supprimer).
- `[[Note|alias]]` garde la cible et non l'alias affiché.
- Le parcours n'est pas récursif, et deux notes de même nom dans des dossiers différents s'écraseraient (clé = `file.name`).
- Un wikilink dans un bloc de code est aussi modifié.
- Pas encore traités : commentaires `%%…%%`, callouts, tags inline, liens Markdown, normalisation Unicode, BOM, CRLF.

**Qualité de code**
- Pas de tests.
- Imports dispersés dans le fichier ; `print` au lieu de `logging`.
- Variable `i` inutilisée dans `enumerate` ; `touch()` inutile avant l'écriture.
- Nom de classe `cleanFile` au lieu de `CleanFile` (convention Python).

> [!tip] Proposition disponible
> Une réécriture complète de `cleaning.py` avec 77 tests a été proposée le 2026-11-01, mais **pas intégrée** : j'ai choisi de garder mon propre code pour progresser. Elle reste une référence pour comparer mes corrections.

## Prochaines étapes

1. Corriger les bugs ci-dessus, un commit par correction.
2. Écrire mes propres tests `pytest` dans `tests/test_cleaning.py`.
3. Commiter `.gitignore` (ajout de `.vscode/`), puis fusionner la branche `feature/nettoyage-markdown` via une Pull Request.
4. Passer au chunking sur une nouvelle branche.
5. Plus tard : traduire ce README en anglais avant de rendre le dépôt public (stage à l'étranger).