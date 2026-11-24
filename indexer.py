"""Incremental indexing: find which notes changed and re-index only those."""

import hashlib

from chunking import chunk_file
from cleaning import NOTE_SUFFIXES, clean_file, env_path, load_env_file
from embedder import embed_passages
from register import get_all_hashes, remove, set_hash
from vectors import delete_by_source, upsert_chunks


def file_hash(path) -> str:
    """Return the SHA-256 of a file's raw bytes, as a hex string (64 characters)."""
    with open(path, "rb") as f:
        digest = hashlib.file_digest(f, "sha256")  # reads the file by blocks (Python 3.11+)
    return digest.hexdigest()


def scan_folder() -> dict:
    """Return {file name: hash} for every note in STORAGE_DIR."""
    scan = {}
    for path in sorted(env_path("STORAGE_DIR").iterdir()):
        if not path.is_file() or path.suffix.lower() not in NOTE_SUFFIXES:
            continue  # skip sub-folders and other file types
        scan[path.name] = file_hash(path)
    return scan


def plan_changes(disk: dict, registry: dict) -> tuple:
    """Compare the folder with the registry.

    disk, registry: {file name: hash}.
    Returns (new, modified, deleted): three sorted lists of file names.
    """
    new = []
    modified = []
    deleted = []

    for name, disk_hash in disk.items():
        previous_hash = registry.get(name)
        if previous_hash is None:
            new.append(name)
        elif previous_hash != disk_hash:
            modified.append(name)

    for name in registry:
        if name not in disk:
            deleted.append(name)
    return sorted(new), sorted(modified), sorted(deleted)


def index_file(name: str) -> None:
    """Clean, chunk, embed and store one note.

    name: the file name inside STORAGE_DIR (e.g. "note.md").
    Does not touch the registry: the caller records the hash after this succeeds.
    """
    path = env_path("STORAGE_DIR") / name
    cleaned = clean_file(path)
    if cleaned is None:
        raise ValueError(f"not a note: {name}")
    chunks = chunk_file(cleaned)
    if not chunks:
        return  # empty note: nothing to embed
    embeddings = embed_passages(chunks)
    upsert_chunks(chunks, embeddings)


def main() -> int:
    """Index only the notes that are new, modified or deleted since the last run.

    Returns the number of notes that failed (0 = everything went well).
    """
    load_env_file()
    disk = scan_folder()
    registry_hashes = get_all_hashes()
    new, modified, deleted = plan_changes(disk, registry_hashes)
    unchanged = len(disk) - len(new) - len(modified)
    failed = []

    # 1. Notes removed from the folder: forget their chunks, then their registry row
    for name in deleted:
        delete_by_source(name)
        remove(name)

    # 2. Notes to (re)index. One failure must not stop the others.
    for name in modified + new:
        try:
            if name in modified:
                delete_by_source(name)  # remove the old version first (no ghost chunks)
            index_file(name)
            set_hash(name, disk[name])  # LAST: only recorded if everything above worked
        except Exception as error:
            failed.append(name)
            print(f"[!] {name}: {error}")  # no hash written: the note is retried next time

    print(f"{len(new)} added, {len(modified)} modified, {len(deleted)} deleted, "
          f"{unchanged} unchanged, {len(failed)} failed")
    return len(failed)


if __name__ == "__main__":
    # A non-zero exit code lets cron or systemd notice that something failed
    raise SystemExit(1 if main() else 0)