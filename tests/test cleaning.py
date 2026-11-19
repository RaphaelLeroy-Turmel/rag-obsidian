"""Tests de cleaning.py.

Principe : ces tests décrivent le comportement ATTENDU du code.
- Les tests normaux passent avec ton code actuel.
- Les tests marqués `xfail(strict=True)` décrivent un comportement voulu que ton
  code n'a pas encore (limite ou bug connu). Ils sont "attendus en échec" :
  la suite reste verte, mais le test documente le problème.
  Quand tu corrigeras le code, pytest signalera XPASS(strict) = échec :
  c'est le signal pour retirer le marqueur xfail de ce test.

Lancer : python -m pytest -v   (depuis la racine du projet)
"""
import os

import pytest

import cleaning
from cleaning import cleanFile, clean_files, clean_wikilinks, load_env_file


# ---------------------------------------------------------------------------
# clean_wikilinks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "entree, attendu",
    [
        # wikilink simple
        ("Voir [[Ridge]] pour plus.", "Voir Ridge pour plus."),
        # avec alias : ton code garde la CIBLE, pas l'alias (choix à confirmer)
        ("Le [[Surapprentissage|surapprentissage]] est un risque.",
         "Le Surapprentissage est un risque."),
        # avec section
        ("Voir [[Ridge#Définition]].", "Voir Ridge."),
        # section + alias
        ("Voir [[Ridge#Définition|la définition]].", "Voir Ridge."),
        # plusieurs wikilinks sur la même ligne (le non-gourmand compte !)
        ("[[A]] puis [[B]] et [[C|c]]", "A puis B et C"),
        # accents et espaces dans le nom
        ("[[Réseau de neurones]]", "Réseau de neurones"),
    ],
)
def test_wikilinks_nominal(entree, attendu):
    assert clean_wikilinks(entree) == attendu


@pytest.mark.parametrize(
    "entree, attendu",
    [
        # Embeds : le "!" est retiré et le NOM du fichier reste dans le texte.
        # (Le commentaire en tête de ton fichier parle de "supprimer" : ici le
        # nom est conservé. Ce test fige le comportement ACTUEL.)
        ("![[schema-l2.png]]", "schema-l2.png"),
        ("Voir ![[Autre note]] ici", "Voir Autre note ici"),
    ],
)
def test_embeds_comportement_actuel(entree, attendu):
    assert clean_wikilinks(entree) == attendu


@pytest.mark.parametrize(
    "texte",
    [
        "",
        "Du texte sans aucun lien.",
        "# Titre\n\nUn paragraphe.\n",
        "Un lien [markdown](http://exemple.com) classique",
        "Des crochets [simples] seulement",
    ],
)
def test_texte_sans_wikilink_inchange(texte):
    assert clean_wikilinks(texte) == texte


def test_wikilinks_multilignes_preserve_les_retours_a_la_ligne():
    entree = "Ligne 1 [[A]]\nLigne 2 [[B|b]]\n"
    assert clean_wikilinks(entree) == "Ligne 1 A\nLigne 2 B\n"


def test_wikilinks_idempotence():
    entree = "Voir [[Ridge#Def|la def]] et ![[img.png]] et [[Lasso]]."
    une_fois = clean_wikilinks(entree)
    assert clean_wikilinks(une_fois) == une_fois


@pytest.mark.xfail(
    strict=True,
    reason="Limite connue : les wikilinks dans un bloc de code sont aussi modifiés",
)
def test_wikilink_dans_bloc_de_code_doit_rester_intact():
    entree = "```python\nx = d[[0]]\n```"
    assert clean_wikilinks(entree) == entree


@pytest.mark.xfail(
    strict=True,
    reason="Limite connue : un wikilink avec cible vide ou seulement un alias "
           "n'est pas géré",
)
def test_wikilink_avec_alias_seul():
    # [[|alias]] n'est pas un cas réaliste, mais montre la limite de la regex
    assert clean_wikilinks("[[|alias]]") == "alias"


# ---------------------------------------------------------------------------
# cleanFile (dataclass)
# ---------------------------------------------------------------------------

def test_cleanfile_champs():
    f = cleanFile("note.md", "contenu", {"tags": ["ml"]})
    assert f.name == "note.md"
    assert f.content == "contenu"
    assert f.metadata == {"tags": ["ml"]}


def test_cleanfile_metadata_par_defaut_vide():
    assert cleanFile("a.md", "x").metadata == {}


def test_cleanfile_metadata_non_partagee_entre_instances():
    # piège classique des valeurs par défaut mutables (default_factory)
    a = cleanFile("a.md", "x")
    b = cleanFile("b.md", "y")
    a.metadata["k"] = "v"
    assert b.metadata == {}


def test_cleanfile_str_contenu_court_non_tronque():
    texte = str(cleanFile("note.md", "court", {"a": 1}))
    assert "Fichier : note.md" in texte
    assert "court" in texte
    assert "..." not in texte


def test_cleanfile_str_contenu_long_tronque_a_400():
    texte = str(cleanFile("note.md", "x" * 1000))
    assert "x" * 400 + "..." in texte
    assert "x" * 401 not in texte


def test_cleanfile_str_exactement_400_non_tronque():
    texte = str(cleanFile("note.md", "x" * 400))
    assert "x" * 400 in texte
    assert "..." not in texte


# ---------------------------------------------------------------------------
# load_env_file
# ---------------------------------------------------------------------------

@pytest.fixture
def env_propre(monkeypatch):
    """load_env_file écrit dans os.environ. On déclare les clés avec
    monkeypatch avant, pour que pytest les restaure à la fin du test."""
    for cle in ("TEST_A", "TEST_B", "TEST_C", "TEST_D"):
        monkeypatch.setenv(cle, "valeur-initiale")
    return monkeypatch


def test_env_fichier_absent_renvoie_dict_vide(tmp_path, capsys):
    assert load_env_file(tmp_path / "inexistant.env") == {}
    assert "ERROR" in capsys.readouterr().out


def test_env_lecture_basique(tmp_path, env_propre):
    f = tmp_path / ".env"
    f.write_text("TEST_A=un\nTEST_B=deux\n", encoding="utf-8")
    assert load_env_file(f) == {"TEST_A": "un", "TEST_B": "deux"}


def test_env_remplit_os_environ(tmp_path, env_propre):
    f = tmp_path / ".env"
    f.write_text("TEST_A=nouvelle\n", encoding="utf-8")
    load_env_file(f)
    assert os.environ["TEST_A"] == "nouvelle"


def test_env_ignore_commentaires_et_lignes_vides(tmp_path, env_propre):
    f = tmp_path / ".env"
    f.write_text("# un commentaire\n\n   \nTEST_A=ok\n  # indenté\n", encoding="utf-8")
    assert load_env_file(f) == {"TEST_A": "ok"}


def test_env_retire_les_guillemets_et_espaces(tmp_path, env_propre):
    f = tmp_path / ".env"
    f.write_text(
        "TEST_A = \"avec guillemets\"\nTEST_B='simples'\nTEST_C =  nu  \n",
        encoding="utf-8",
    )
    assert load_env_file(f) == {
        "TEST_A": "avec guillemets",
        "TEST_B": "simples",
        "TEST_C": "nu",
    }


def test_env_valeur_contenant_un_egal(tmp_path, env_propre):
    # split("=", 1) : seul le premier "=" sépare clé et valeur
    f = tmp_path / ".env"
    f.write_text("TEST_A=abc=def==\n", encoding="utf-8")
    assert load_env_file(f)["TEST_A"] == "abc=def=="


def test_env_ligne_sans_egal_leve_valueerror(tmp_path, env_propre):
    # Comportement actuel : une ligne mal formée fait planter la fonction.
    f = tmp_path / ".env"
    f.write_text("PASDEGAL\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_env_file(f)


def test_env_valeur_vide(tmp_path, env_propre):
    f = tmp_path / ".env"
    f.write_text("TEST_A=\n", encoding="utf-8")
    assert load_env_file(f) == {"TEST_A": ""}


# ---------------------------------------------------------------------------
# clean_files
# ---------------------------------------------------------------------------

@pytest.fixture
def dossiers(tmp_path, monkeypatch):
    """Crée un dossier de notes et un dossier de sortie temporaires, et
    pointe STORAGE_DIR / CLEAN_DIR dessus."""
    notes = tmp_path / "notes"
    sortie = tmp_path / "clean"
    notes.mkdir()
    sortie.mkdir()
    monkeypatch.setenv("STORAGE_DIR", str(notes))
    monkeypatch.setenv("CLEAN_DIR", str(sortie))
    return notes, sortie


NOTE_EXEMPLE = """---
tags: [ml, cours]
date: 2026-10-12
---
# Régularisation

Le [[Surapprentissage|surapprentissage]] est limité.
Voir aussi [[Ridge]] et ![[schema-l2.png]].
"""


def test_clean_files_dossier_vide(dossiers):
    assert clean_files() == {}


def test_clean_files_ignore_les_fichiers_non_md(dossiers):
    notes, _ = dossiers
    (notes / "cours.pdf").write_bytes(b"%PDF-1.4")
    (notes / "image.png").write_bytes(b"\x89PNG")
    (notes / "notes.txt").write_text("texte", encoding="utf-8")
    assert clean_files() == {}


def test_clean_files_sans_storage_dir_leve_typeerror(monkeypatch):
    # Path(None) : la variable d'environnement manque -> erreur peu explicite
    monkeypatch.delenv("STORAGE_DIR", raising=False)
    monkeypatch.setenv("CLEAN_DIR", "quelque_part")
    with pytest.raises(TypeError):
        clean_files()


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="Bug connu : clean_file.write_text() est appelé sans contenu "
           "(TypeError), donc la fonction plante dès le premier .md",
)
def test_clean_files_nettoie_une_note(dossiers):
    notes, _ = dossiers
    (notes / "regul.md").write_text(NOTE_EXEMPLE, encoding="utf-8")

    resultat = clean_files()

    assert list(resultat) == ["regul.md"]
    doc = resultat["regul.md"]
    assert isinstance(doc, cleanFile)
    assert doc.metadata["tags"] == ["ml", "cours"]
    assert "---" not in doc.content            # frontmatter retiré du corps
    assert "[[" not in doc.content             # wikilinks nettoyés
    assert "Le Surapprentissage est limité." in doc.content


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="Même bug que ci-dessus (write_text sans argument)",
)
def test_clean_files_ecrit_un_fichier_nettoye_dans_clean_dir(dossiers):
    notes, sortie = dossiers
    (notes / "regul.md").write_text(NOTE_EXEMPLE, encoding="utf-8")

    clean_files()

    # Comportement voulu : un fichier de sortie existe dans CLEAN_DIR
    fichiers = list(sortie.iterdir())
    assert len(fichiers) == 1
    assert "Ridge" in fichiers[0].read_text(encoding="utf-8")


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="Même bug (write_text sans argument)",
)
def test_clean_files_note_sans_frontmatter(dossiers):
    notes, _ = dossiers
    (notes / "simple.md").write_text("# Titre\n\nTexte [[A]].\n", encoding="utf-8")

    doc = clean_files()["simple.md"]

    assert doc.metadata == {}
    assert doc.content == "# Titre\n\nTexte A."


# ---------------------------------------------------------------------------
# main / import
# ---------------------------------------------------------------------------

def test_import_n_execute_pas_main(capsys):
    # Le garde `if __name__ == "__main__"` doit éviter tout affichage à l'import
    import importlib
    importlib.reload(cleaning)
    assert capsys.readouterr().out == ""