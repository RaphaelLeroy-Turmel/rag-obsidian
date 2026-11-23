# run_eval.py : measure the retrieval quality of the RAG on a fixed set of questions
#
# Run from the root of the repository (so that cleaning, chunking and embedder can be imported):
#     python -m eval.run_eval
#
# Two levels of evaluation for each question:
#   - SOURCE level : is a chunk from the right document in the top k?
#   - CHUNK level  : is a chunk from the right document AND containing the evidence
#                    (a short sentence from the passage with the answer) in the top k?
# Metrics: recall@k (k = 1, 3, 5, 10) and MRR (Mean Reciprocal Rank).
#
# At the end, a regression check compares the LanceDB search (vectors.py) with the
# in-memory search: the ranks of the first relevant chunk must be identical.

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from chunking import MODEL_NAME, MAX_TOKENS, chunk_files
from cleaning import clean_files, env_path, load_env_file
from embedder import embed_passages, embed_query, get_best_embedding, get_model
from vectors import get_table, search, upsert_chunks

# --- Settings ---------------------------------------------------------------

EVAL_DIR = Path(__file__).parent                  # the eval/ folder
QUESTIONS_FILE = EVAL_DIR / "questions.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
K_VALUES = [1, 3, 5, 10]                          # the k used for recall@k
TOP_K = max(K_VALUES)                             # number of chunks retrieved per question


# --- Loading ----------------------------------------------------------------

def load_questions(path: Path) -> list[dict]:
    """Read the JSONL file: one question (a dict) per line, empty lines ignored."""
    questions = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def load_or_compute_embeddings(chunks: list, cache_file: Path) -> np.ndarray:
    """Return the chunk embeddings, reusing a cache file if the chunks did not change.

    The cache stores the chunk ids next to the vectors. chunk_id is a hash of the
    chunk text, so if any chunk changes (new note, new chunking settings), the ids
    differ and the embeddings are recomputed.
    """
    ids = [chunk.chunk_id for chunk in chunks]
    if cache_file.exists():
        cache = np.load(cache_file, allow_pickle=False)
        same_model = str(cache["model"]) == MODEL_NAME
        same_chunks = cache["ids"].tolist() == ids
        if same_model and same_chunks:
            print(f"Embeddings loaded from cache ({len(ids)} chunks)")
            return cache["vectors"]
    print(f"Computing embeddings for {len(ids)} chunks (this can take a few minutes)...")
    vectors = embed_passages(chunks)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, vectors=vectors, ids=np.array(ids), model=np.array(MODEL_NAME))
    return vectors


# --- Text matching ----------------------------------------------------------

def normalize(text: str) -> str:
    """Lowercase and clean a text so that small PDF extraction differences do not matter.

    - "differ-\\nent" (word cut at the end of a PDF line) becomes "different"
    - curly quotes become straight quotes
    - "_" and "-" between words do not matter for file names (handled in source_matches)
    - all whitespace (spaces, new lines, tabs) becomes a single space
    """
    text = text.lower()
    text = text.replace("-\n", "")                      # hyphenation at the end of a PDF line
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    return " ".join(text.split())                       # split() cuts on any whitespace


def source_matches(source: str, expected_sources: list[str]) -> bool:
    """True if the file name contains one of the expected keys (case and _ - ignored)."""
    name = normalize(source.replace("_", " ").replace("-", " "))
    for key in expected_sources:
        if normalize(key.replace("-", " ")) in name:
            return True
    return False


def chunk_is_relevant(chunk, question: dict) -> bool:
    """True if the chunk comes from an expected source AND contains one evidence phrase."""
    if not source_matches(chunk.source, question["expected_sources"]):
        return False
    text = normalize(chunk.text)
    for phrase in question["evidence"]:
        if normalize(phrase) in text:
            return True
    return False


# --- Metrics ----------------------------------------------------------------

def first_rank(flags: list[bool]):
    """Rank (starting at 1) of the first True in the list, or None if there is none."""
    for position, flag in enumerate(flags, start=1):
        if flag:
            return position
    return None


def summarize(ranks: list) -> dict:
    """Compute recall@k and MRR from a list of ranks (None = not found in the top k)."""
    n = len(ranks)
    if n == 0:
        return {"n": 0}
    summary = {"n": n}
    for k in K_VALUES:
        found = 0
        for rank in ranks:
            if rank is not None and rank <= k:
                found += 1
        summary[f"recall@{k}"] = round(found / n, 3)
    reciprocal_ranks = []
    for rank in ranks:
        if rank is None:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(1 / rank)
    summary["mrr"] = round(sum(reciprocal_ranks) / n, 3)
    return summary


def format_summary(name: str, summary: dict) -> str:
    """One line of the results table."""
    if summary["n"] == 0:
        return f"{name:<22} (no question)"
    recalls = "  ".join(f"R@{k} {summary[f'recall@{k}']:.2f}" for k in K_VALUES)
    return f"{name:<22} n={summary['n']:<3} {recalls}  MRR {summary['mrr']:.2f}"


# --- Regression check: LanceDB vs in-memory search ---------------------------

def check_lancedb_regression(chunks: list, vectors: np.ndarray, questions: list[dict]) -> None:
    """Check that the LanceDB search gives the same ranks as the in-memory search.

    chunks, vectors: the corpus and its embeddings (same order); questions: the eval questions.
    Writes the chunks to LanceDB, then runs every question through both searches.
    The exact order of two chunks with the same score is not guaranteed, so a different
    top-k list is only reported; what must stay identical is the rank of the first
    relevant chunk (source level and chunk level), because the metrics depend on it.
    """
    upsert_chunks(chunks, vectors)
    table = get_table()
    print(f"\nRows in table: {table.count_rows()} | chunks in corpus: {len(chunks)}")

    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    mismatches = 0
    rank_mismatches = 0

    for q in questions:
        query_vector = embed_query(q["question"])
        memory_indices = get_best_embedding(vectors, TOP_K, query_vector)
        memory_ids = [chunks[i].chunk_id for i in memory_indices]
        lance_rows = search(query_vector, k=TOP_K)
        lance_ids = [row["id"] for row in lance_rows]

        # Same ids in the same order? (can differ when two chunks have the same score)
        if memory_ids != lance_ids:
            mismatches += 1
            print(f"[!] {q['id']}: top-{TOP_K} differs")
            for rank in range(TOP_K):
                memory_score = float(vectors[memory_indices[rank]] @ query_vector)
                lance_score = lance_rows[rank]["score"]
                print(f"    {rank + 1:>2}  memory {memory_ids[rank]} {memory_score:.6f}"
                      f"  | lance {lance_ids[rank]} {lance_score:.6f}")

        # Same rank of the first relevant chunk? (this is what the metrics use)
        if q["type"] != "negative":
            memory_chunks = [chunks[i] for i in memory_indices]
            lance_chunks = [chunk_by_id[chunk_id] for chunk_id in lance_ids]
            memory_ranks = (
                first_rank([source_matches(c.source, q["expected_sources"]) for c in memory_chunks]),
                first_rank([chunk_is_relevant(c, q) for c in memory_chunks]),
            )
            lance_ranks = (
                first_rank([source_matches(c.source, q["expected_sources"]) for c in lance_chunks]),
                first_rank([chunk_is_relevant(c, q) for c in lance_chunks]),
            )
            if memory_ranks != lance_ranks:
                rank_mismatches += 1
                print(f"[!] {q['id']}: ranks differ (source, chunk): {memory_ranks} vs {lance_ranks}")

    print(f"Questions with a different top-{TOP_K}: {mismatches} / {len(questions)}")
    print(f"Questions with different ranks: {rank_mismatches} / {len(questions)}")


# --- Main -------------------------------------------------------------------

def main():
    load_env_file()
    get_model()     # load the model first, so that embed_query() works even with a cached index

    questions = load_questions(QUESTIONS_FILE)
    chunks = chunk_files(clean_files())
    vectors = load_or_compute_embeddings(chunks, env_path("CLEAN_DIR") / "embeddings_cache.npz")
    print(f"{len(questions)} questions | {len(chunks)} chunks | vectors {vectors.shape}\n")

    results = []
    for q in questions:
        query_vector = embed_query(q["question"])
        scores = vectors @ query_vector                       # cosine similarity with every chunk
        best_indices = get_best_embedding(vectors, TOP_K, query_vector)
        best_chunks = [chunks[i] for i in best_indices]

        result = {
            "id": q["id"],
            "type": q["type"],
            "lang": q["lang"],
            "top1_score": round(float(scores[best_indices[0]]), 3),
            "top1_source": best_chunks[0].source,
            "top_sources": [c.source for c in best_chunks],
        }

        if q["type"] != "negative":
            source_flags = [source_matches(c.source, q["expected_sources"]) for c in best_chunks]
            chunk_flags = [chunk_is_relevant(c, q) for c in best_chunks]
            result["source_rank"] = first_rank(source_flags)
            result["chunk_rank"] = first_rank(chunk_flags)
            # Sanity check of the question itself: does the evidence exist in the corpus at all?
            relevant_in_corpus = 0
            for c in chunks:
                if chunk_is_relevant(c, q):
                    relevant_in_corpus += 1
            result["relevant_chunks_in_corpus"] = relevant_in_corpus

        results.append(result)

    # --- Per-question table ---
    print(f"{'id':<9} {'type':<14} {'src':>4} {'chunk':>5} {'score':>6}  top-1 source")
    for r in results:
        if r["type"] == "negative":
            src, chk = "-", "-"
        else:
            src = r["source_rank"] if r["source_rank"] is not None else "miss"
            chk = r["chunk_rank"] if r["chunk_rank"] is not None else "miss"
        print(f"{r['id']:<9} {r['type']:<14} {src!s:>4} {chk!s:>5} {r['top1_score']:>6.3f}  {r['top1_source'][:45]}")

    # --- Warnings about the questions themselves ---
    positives = [r for r in results if r["type"] != "negative"]
    no_evidence = [r for r in positives if r["relevant_chunks_in_corpus"] == 0]
    if no_evidence:
        print("\n[!] Evidence found in NO chunk of the corpus (fix the question, or the source is missing):")
        for r in no_evidence:
            print(f"    {r['id']}")
    # Chunk-level metrics only make sense when the evidence exists somewhere
    checkable = [r for r in positives if r["relevant_chunks_in_corpus"] > 0]

    # --- Aggregated metrics ---
    report = {
        "source_level": summarize([r["source_rank"] for r in positives]),
        "chunk_level": summarize([r["chunk_rank"] for r in checkable]),
        "by_type": {},
        "by_lang": {},
    }
    for question_type in sorted({r["type"] for r in positives}):
        subset = [r["source_rank"] for r in positives if r["type"] == question_type]
        report["by_type"][question_type] = summarize(subset)
    for lang in sorted({r["lang"] for r in positives}):
        subset = [r["source_rank"] for r in positives if r["lang"] == lang]
        report["by_lang"][lang] = summarize(subset)

    print("\n=== Overall ===")
    print(format_summary("source level", report["source_level"]))
    print(format_summary("chunk level", report["chunk_level"]))
    print("\n=== Source level, by question type ===")
    for name, summary in report["by_type"].items():
        print(format_summary(name, summary))
    print("\n=== Source level, by language ===")
    for name, summary in report["by_lang"].items():
        print(format_summary(name, summary))

    # --- Negative questions: is the best score clearly lower than for real questions? ---
    negatives = [r for r in results if r["type"] == "negative"]
    if negatives and positives:
        mean_positive = np.mean([r["top1_score"] for r in positives])
        mean_negative = np.mean([r["top1_score"] for r in negatives])
        report["top1_score_mean"] = {"positive": round(float(mean_positive), 3),
                                     "negative": round(float(mean_negative), 3)}
        print("\n=== Top-1 score: questions with an answer vs without ===")
        print(f"with an answer    : {mean_positive:.3f}")
        print(f"without an answer : {mean_negative:.3f}")
        print("(a small gap means a score threshold cannot easily detect 'I don't know')")

    # --- Save everything, to compare later runs ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / f"eval_{datetime.now():%Y%m%d_%H%M}.json"
    config = {"model": MODEL_NAME, "max_tokens": MAX_TOKENS, "n_chunks": len(chunks), "top_k": TOP_K}
    with out_file.open("w", encoding="utf-8") as f:
        json.dump({"config": config, "report": report, "questions": results}, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_file}")

    check_lancedb_regression(chunks, vectors, questions)


if __name__ == "__main__":
    main()