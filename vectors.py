"""Vector storage with LanceDB: save chunk embeddings and search them."""

import lancedb
import numpy as np
import pyarrow as pa

from chunking import Chunk
from cleaning import env_path, load_env_file

TABLE_NAME = "chunks"
EMBEDDING_DIM = 384  # size of the multilingual-e5-small vectors

_db = None  # lazy-loaded connection, same pattern as get_model()


def get_table():
    """Open the chunks table, creating it (empty) if it does not exist yet."""
    global _db
    if _db is None:
        _db = lancedb.connect(str(env_path("DB_DIR")))
    if TABLE_NAME in _db.list_tables():
        return _db.open_table(TABLE_NAME)
    schema = pa.schema([
        pa.field("id", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), EMBEDDING_DIM)),
        pa.field("text", pa.string()),
        pa.field("source", pa.string()),
        pa.field("headings", pa.string()),
    ])
    return _db.create_table(TABLE_NAME, schema=schema)


def upsert_chunks(chunks: list[Chunk], vectors) -> None:
    """Insert new chunks or update existing ones (matched on chunk id).

    chunks: list of Chunk; vectors: array of shape (len(chunks), EMBEDDING_DIM).
    """
    if len(chunks) != len(vectors):
        raise ValueError(
            f"chunks and vectors differ in length: {len(chunks)} vs {len(vectors)}"
        )
    if not chunks:
        return  # nothing to write

    rows = []
    for chunk, vector in zip(chunks, vectors):
        row = {
            "id": chunk.chunk_id,
            "vector": vector,
            "text": chunk.text,  # original text, without the "passage: " prefix
            "source": chunk.source,
            "headings": " > ".join(chunk.headings),
        }
        rows.append(row)

    table = get_table()
    table.merge_insert("id").when_matched_update_all().when_not_matched_insert_all().execute(rows)


def delete_by_source(source: str) -> None:
    """Delete every chunk that comes from one source file.

    source: the source name stored in the table (e.g. "note.md").
    """
    if not source:
        raise ValueError("source is empty")
    # The filter is an SQL string: a quote inside the name must be doubled
    safe_source = source.replace("'", "''")
    sql_filter = f"source = '{safe_source}'"
    get_table().delete(sql_filter)


def search(query_vector, k: int = 10) -> list[dict]:
    """Return the k closest chunks, best first, with a similarity score.

    query_vector: array of shape (EMBEDDING_DIM,); k: number of chunks wanted.
    Each result is a dict with id, text, source, headings and score (cosine, max 1.0).
    """
    if len(query_vector) != EMBEDDING_DIM:
        raise ValueError(
            f"query vector has {len(query_vector)} values, expected {EMBEDDING_DIM}"
        )
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    raw_results = get_table().search(query_vector).metric("cosine").limit(k).to_list()

    results = []
    for raw in raw_results:
        result = {
            "id": raw["id"],
            "text": raw["text"],
            "source": raw["source"],
            "headings": raw["headings"],
            "score": 1 - raw["_distance"],  # LanceDB returns 1 - cosine
        }
        results.append(result)
    return results


def make_test_vectors(count: int, seed: int = 42):
    """Return `count` random unit vectors (float32), shape (count, EMBEDDING_DIM)."""
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((count, EMBEDDING_DIM)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms


def count_source(source: str) -> int:
    """Count the rows of one source (test helper)."""
    safe_source = source.replace("'", "''")
    return get_table().count_rows(f"source = '{safe_source}'")


if __name__ == "__main__":
    load_env_file()
    table = get_table()
    print("Rows at start:", table.count_rows())

    # 0. Clean leftovers of a previous run (also tests delete on a missing source)
    delete_by_source("test.md")
    delete_by_source("it's.md")
    start_count = get_table().count_rows()

    # 1. Insert 3 chunks
    chunks = [
        Chunk(text="first", source="test.md", headings=["A", "B"], chunk_id="t1"),
        Chunk(text="second", source="test.md", headings=[], chunk_id="t2"),
        Chunk(text="third", source="test.md", headings=[], chunk_id="t3"),
    ]
    vectors = make_test_vectors(3)
    upsert_chunks(chunks, vectors)
    assert get_table().count_rows() == start_count + 3
    print("1. insert OK")

    # 2. Same ids again: no duplicate
    upsert_chunks(chunks, vectors)
    assert get_table().count_rows() == start_count + 3
    print("2. no duplicate OK")

    # 3. Same id, new text: the row is updated, not duplicated
    chunks[0].text = "first (modified)"
    upsert_chunks(chunks, vectors)
    assert get_table().count_rows() == start_count + 3
    best = search(vectors[0], k=1)[0]
    assert best["text"] == "first (modified)"
    print("3. update OK")

    # 4. Search: the closest chunk to vector i is chunk i, with score ~ 1
    for i in range(3):
        top = search(vectors[i], k=3)
        assert top[0]["id"] == chunks[i].chunk_id
        assert top[0]["score"] > 0.999
        assert top[0]["score"] >= top[1]["score"] >= top[2]["score"]
    assert search(vectors[0], k=1)[0]["headings"] == "A > B"
    assert len(search(vectors[0], k=2)) == 2
    print("4. search OK")

    # 5. Bad arguments
    for bad_call in (
        lambda: search(vectors[0], k=0),
        lambda: search(vectors[0][:10]),
        lambda: delete_by_source(""),
        lambda: upsert_chunks(chunks, vectors[:2]),
    ):
        try:
            bad_call()
        except ValueError:
            pass
        else:
            raise AssertionError("ValueError expected")
    print("5. errors OK")

    # 6. Delete by source, with an apostrophe in the name
    quote_chunk = Chunk(text="quote", source="it's.md", headings=[], chunk_id="q1")
    upsert_chunks([quote_chunk], make_test_vectors(1, seed=7))
    assert count_source("it's.md") == 1
    delete_by_source("it's.md")
    assert count_source("it's.md") == 0
    assert count_source("test.md") == 3  # other sources untouched
    print("6. delete with apostrophe OK")

    # 7. Cleanup
    delete_by_source("test.md")
    assert get_table().count_rows() == start_count
    print("7. cleanup OK, rows at end :", get_table().count_rows())