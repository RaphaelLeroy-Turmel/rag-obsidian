"""SQLite registry: remember which files are indexed and the hash they had."""

import sqlite3
from datetime import datetime

from cleaning import env_path, load_env_file

REGISTRY_TABLE = "files"  # not the LanceDB table name ("chunks"): two different databases

_con = None  # lazy-loaded connection, same pattern as get_table()


def get_connection():
    """Open the registry database once, creating the file and the table if needed."""
    global _con
    if _con is None:
        db_file = env_path("DB_DIR") / "registry.sqlite"
        db_file.parent.mkdir(parents=True, exist_ok=True)  # connect() fails if the folder is missing
        _con = sqlite3.connect(str(db_file))
        _con.execute(
            f"CREATE TABLE IF NOT EXISTS {REGISTRY_TABLE} ("
            "path TEXT PRIMARY KEY, "        # one row per file
            "content_hash TEXT NOT NULL, "
            "indexed_at TEXT NOT NULL)"
        )
        _con.commit()
    return _con


def get_all_hashes() -> dict:
    """Return every indexed file as a dict {path: content_hash}."""
    con = get_connection()
    rows = con.execute(f"SELECT path, content_hash FROM {REGISTRY_TABLE}").fetchall()
    hashes = {path: content_hash for path, content_hash in rows}
    return hashes


def set_hash(path: str, content_hash: str) -> None:
    """Remember that `path` was indexed with this hash (insert, or update if it exists)."""
    con = get_connection()
    indexed_at = datetime.now().isoformat(timespec="seconds")
    # The ? placeholders let SQLite handle quotes in the values (no f-string for values)
    con.execute(
        f"INSERT OR REPLACE INTO {REGISTRY_TABLE} (path, content_hash, indexed_at) "
        "VALUES (?, ?, ?)",
        (path, content_hash, indexed_at),
    )
    con.commit()  # without commit, the change is lost when the script ends


def remove(path: str) -> None:
    """Forget a file (it was deleted from the notes folder)."""
    con = get_connection()
    con.execute(f"DELETE FROM {REGISTRY_TABLE} WHERE path = ?", (path,))  # (path,) = a tuple of 1
    con.commit()


if __name__ == "__main__":
    load_env_file()
    start = get_all_hashes()
    print("Files in registry at start:", len(start))

    # Clean leftovers of a previous run
    remove("test_a.md")
    remove("it's.md")

    set_hash("test_a.md", "hash1")
    set_hash("it's.md", "hash2")                      # apostrophe in the name
    hashes = get_all_hashes()
    assert hashes["test_a.md"] == "hash1" and hashes["it's.md"] == "hash2"
    assert len(hashes) == len(start) + 2
    print("1. insert OK")

    set_hash("test_a.md", "hash1-modified")           # same path: update, no duplicate
    hashes = get_all_hashes()
    assert hashes["test_a.md"] == "hash1-modified"
    assert len(hashes) == len(start) + 2
    print("2. update OK")

    remove("it's.md")
    remove("it's.md")                                 # removing twice must not fail
    remove("does_not_exist.md")
    assert "it's.md" not in get_all_hashes()
    print("3. remove OK")

    remove("test_a.md")
    assert get_all_hashes() == start
    print("4. cleanup OK")