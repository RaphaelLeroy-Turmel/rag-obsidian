"""Integration tests of the incremental indexing, on temporary folders (never your real data).

Run from the repository root:
    python -m pytest tests/test_incremental.py -q

The fast tests use fake embeddings (no model needed). One slow test uses the real model:
    $env:RUN_SLOW = "1"
    python -m pytest tests/test_incremental.py -q
"""

import hashlib
import os
from types import SimpleNamespace

import numpy as np
import pytest

import indexer
import register
import vectors
from cleaning import clean_file
from vectors import EMBEDDING_DIM, get_table


# --- Fixtures: an isolated workspace for every test --------------------------

@pytest.fixture
def bare_workspace(tmp_path, monkeypatch):
    """Temporary notes folder, vector database and register. Nothing real is touched."""
    notes = tmp_path / "notes"
    notes.mkdir()
    monkeypatch.setenv("STORAGE_DIR", str(notes))
    monkeypatch.setenv("CLEAN_DIR", str(tmp_path / "clean"))
    monkeypatch.setenv("DB_DIR", str(tmp_path / "db"))
    # No real .env can be loaded: we work in an empty folder, and load_env_file is disabled
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(indexer, "load_env_file", lambda *args, **kwargs: {}, raising=False)
    # The connections are cached in module variables: forget the ones of a previous test
    monkeypatch.setattr(vectors, "_db", None)
    monkeypatch.setattr(register, "_con", None)
    yield SimpleNamespace(notes=notes)
    if register._con is not None:
        register._con.close()


@pytest.fixture
def workspace(bare_workspace, monkeypatch):
    """Same workspace, with fake embeddings: fast, and no model download."""
    calls = []  # one entry per call of embed_passages (number of chunks)

    def fake_embed_passages(chunks):
        calls.append(len(chunks))
        vectors_list = []
        for chunk in chunks:
            if "BOOM" in chunk.text:
                raise RuntimeError("simulated embedding failure")
            seed = int(hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:8], 16)
            vector = np.random.default_rng(seed).standard_normal(EMBEDDING_DIM)
            vector = vector.astype(np.float32)
            vectors_list.append(vector / np.linalg.norm(vector))
        return np.array(vectors_list)

    monkeypatch.setattr(indexer, "embed_passages", fake_embed_passages)
    bare_workspace.embed_calls = calls
    return bare_workspace


# --- Helpers -----------------------------------------------------------------

def write_note(workspace, name, body):
    """Create a Markdown note with a YAML header in the temporary notes folder."""
    text = f"---\ntags: [test]\n---\n# Title\n\n{body}\n"
    (workspace.notes / name).write_text(text, encoding="utf-8")


def rows_of(name):
    """Every row of the vector table that comes from one source file."""
    safe_name = name.replace("'", "''")
    return get_table().search().where(f"source = '{safe_name}'").limit(1000).to_list()


def run_main(capsys):
    """Run the indexer and return what it printed."""
    capsys.readouterr()  # forget the output of earlier steps
    indexer.main()
    return capsys.readouterr().out


# --- clean_file ---------------------------------------------------------------

def test_clean_file_markdown(bare_workspace):
    write_note(bare_workspace, "a.md", "Hello world")
    cleaned = clean_file(bare_workspace.notes / "a.md")
    assert cleaned.name == "a.md"
    assert "Hello world" in cleaned.content
    assert "tags:" not in cleaned.content            # the YAML header is removed from the text
    assert cleaned.metadata.get("tags") == ["test"]  # ... and kept as metadata


def test_clean_file_suffix_is_case_insensitive(bare_workspace):
    write_note(bare_workspace, "B.MD", "Upper case suffix")
    cleaned = clean_file(bare_workspace.notes / "B.MD")
    assert cleaned is not None and cleaned.name == "B.MD"


def test_clean_file_returns_none_for_other_things(bare_workspace):
    (bare_workspace.notes / "c.txt").write_text("not a note", encoding="utf-8")
    (bare_workspace.notes / "folder.md").mkdir()     # a folder that looks like a note
    assert clean_file(bare_workspace.notes / "c.txt") is None
    assert clean_file(bare_workspace.notes / "folder.md") is None


# --- index_file ---------------------------------------------------------------

def test_index_file_uses_the_file_name_as_source(workspace):
    write_note(workspace, "a.md", "Apples are red.")
    indexer.index_file("a.md")
    rows = rows_of("a.md")
    assert len(rows) >= 1
    # every row of the table has this source: delete_by_source("a.md") will find them all
    assert get_table().count_rows() == len(rows)


def test_index_file_twice_creates_no_duplicate(workspace):
    write_note(workspace, "a.md", "Apples are red.")
    indexer.index_file("a.md")
    first_count = get_table().count_rows()
    indexer.index_file("a.md")
    assert get_table().count_rows() == first_count


def test_index_file_rejects_something_that_is_not_a_note(workspace):
    (workspace.notes / "c.txt").write_text("not a note", encoding="utf-8")
    with pytest.raises(ValueError):
        indexer.index_file("c.txt")


def test_index_file_empty_note_does_not_crash(workspace):
    (workspace.notes / "empty.md").write_text("", encoding="utf-8")
    indexer.index_file("empty.md")                   # must not raise


# --- main: the whole incremental cycle ----------------------------------------

def test_main_first_run_indexes_everything(workspace, capsys):
    write_note(workspace, "a.md", "Apples are red.")
    write_note(workspace, "b.md", "Bananas are yellow.")
    out = run_main(capsys)
    assert "2 added" in out and "0 failed" in out
    assert len(rows_of("a.md")) >= 1 and len(rows_of("b.md")) >= 1
    assert register.get_all_hashes() == indexer.scan_folder()   # the register knows both files


def test_main_second_run_does_nothing(workspace, capsys):
    write_note(workspace, "a.md", "Apples are red.")
    write_note(workspace, "b.md", "Bananas are yellow.")
    run_main(capsys)
    rows_before = get_table().count_rows()
    workspace.embed_calls.clear()

    out = run_main(capsys)
    assert "0 added, 0 modified, 0 deleted" in out
    assert "2 unchanged" in out
    assert workspace.embed_calls == []               # the model was not even used
    assert get_table().count_rows() == rows_before


def test_main_modified_note_replaces_its_old_chunks(workspace, capsys):
    write_note(workspace, "a.md", "Apples are red.")
    write_note(workspace, "b.md", "Bananas are yellow.")
    run_main(capsys)
    b_rows_before = len(rows_of("b.md"))
    old_hash = register.get_all_hashes()["a.md"]

    write_note(workspace, "a.md", "Oranges are orange.")
    out = run_main(capsys)

    assert "0 added, 1 modified, 0 deleted" in out
    texts = [row["text"] for row in rows_of("a.md")]
    assert any("Oranges" in text for text in texts)          # the new text is there
    assert not any("Apples" in text for text in texts)       # no ghost chunk of the old version
    assert len(rows_of("b.md")) == b_rows_before             # the other note is untouched
    assert register.get_all_hashes()["a.md"] != old_hash     # the register has the new hash


def test_main_deleted_note_loses_its_chunks_and_its_register_row(workspace, capsys):
    write_note(workspace, "a.md", "Apples are red.")
    write_note(workspace, "b.md", "Bananas are yellow.")
    run_main(capsys)

    (workspace.notes / "a.md").unlink()
    out = run_main(capsys)

    assert "0 added, 0 modified, 1 deleted" in out
    assert rows_of("a.md") == []
    assert "a.md" not in register.get_all_hashes()
    assert len(rows_of("b.md")) >= 1


def test_main_renamed_note_moves_its_chunks(workspace, capsys):
    write_note(workspace, "old_name.md", "Apples are red.")
    run_main(capsys)

    (workspace.notes / "old_name.md").rename(workspace.notes / "new_name.md")
    out = run_main(capsys)

    assert "1 added, 0 modified, 1 deleted" in out
    assert rows_of("old_name.md") == []
    assert len(rows_of("new_name.md")) >= 1


def test_main_failed_note_is_not_registered_and_is_retried(workspace, capsys):
    write_note(workspace, "good.md", "Apples are red.")
    write_note(workspace, "bad.md", "BOOM this note makes the embedding fail.")

    out = run_main(capsys)
    assert "1 failed" in out
    assert len(rows_of("good.md")) >= 1                      # one failure does not stop the others
    assert "bad.md" not in register.get_all_hashes()         # no hash: it will be retried

    write_note(workspace, "bad.md", "This note is now fine.")
    out = run_main(capsys)
    assert "1 added" in out and "0 failed" in out            # still seen as new: retried
    assert len(rows_of("bad.md")) >= 1


def test_main_empty_note_is_registered_and_not_retried(workspace, capsys):
    (workspace.notes / "empty.md").write_text("", encoding="utf-8")
    run_main(capsys)
    assert "empty.md" in register.get_all_hashes()
    out = run_main(capsys)
    assert "0 added, 0 modified, 0 deleted" in out


def test_main_note_name_with_an_apostrophe(workspace, capsys):
    name = "Notes d'Alice.md"
    write_note(workspace, name, "Apples are red.")
    run_main(capsys)
    assert len(rows_of(name)) >= 1
    assert name in register.get_all_hashes()

    write_note(workspace, name, "Pears are green.")          # modified: delete + re-index
    run_main(capsys)
    texts = [row["text"] for row in rows_of(name)]
    assert any("Pears" in text for text in texts)
    assert not any("Apples" in text for text in texts)

    (workspace.notes / name).unlink()                        # deleted
    run_main(capsys)
    assert rows_of(name) == []
    assert name not in register.get_all_hashes()


# --- Real model (slow, optional) ------------------------------------------------

@pytest.mark.skipif(os.getenv("RUN_SLOW") != "1", reason="set RUN_SLOW=1 to use the real model")
def test_real_model_finds_the_indexed_note(bare_workspace, capsys):
    from embedder import embed_query, get_model

    write_note(bare_workspace, "cats.md",
               "Cats are small carnivorous mammals. They purr, hunt mice and sleep a lot.")
    write_note(bare_workspace, "rockets.md",
               "A rocket engine burns fuel and oxidizer to produce thrust in space.")
    run_main(capsys)

    get_model()
    results = vectors.search(embed_query("Which animal purrs and catches mice?"), k=1)
    assert results[0]["source"] == "cats.md"