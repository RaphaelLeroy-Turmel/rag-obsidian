---
tags: [projet/rag-obsidian, readme]
created: 2026-11-01
updated: 2026-11-22
status: in progress — LanceDB storage written and tested, regression check against the baseline next
---

# RAG Obsidian

A lightweight RAG (Retrieval-Augmented Generation) system that indexes a folder of my Obsidian notes and PDFs, and answers questions while citing the source notes.

## Goal

- Ask questions in natural language over a dedicated folder of notes (`.md` and `.pdf`, in French and English).
- Automatically index new files.
- Final target: run on my homelab (less than 4 GB of available RAM). **For now, all development happens on my PC.**

## Architecture

```
Notes folder (.md, .pdf)
        │
        ▼
  1. Cleaning (+ Unicode NFKC)        (cleaning.py)  ✔
  2. Chunking (by Markdown headings)  (chunking.py)  ✔
  3. Embeddings (multilingual-e5-small, CPU)  (embedder.py)  ✔
     Retrieval evaluation             (eval/)        ✔ baseline
  4. Vector store (LanceDB)           (vectors.py)   ✔ written and tested
                                                     ← next: regression check vs the baseline
  5. Incremental indexing (hash + SQLite registry)
        │
        ▼
  Question → passage retrieval → LLM via API → answer + sources
```

Main design choices: home-made pipeline (no LangChain/LlamaIndex, to understand each step), local embeddings so notes never leave the machine during indexing, embedded vector database (no extra service), LLM through an API because there is no RAM for a local one.

## Retrieval baseline

Measured on 7 scientific papers/books and a few course notes (567 chunks), with 24 questions that have an answer in the corpus and 3 that do not. This is the in-memory search of `embedder.py`; the LanceDB search must reproduce it exactly.

| Level | R@1 | R@3 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| **Source** (a chunk from the right document) | 0.917 | 0.917 | 0.917 | 0.958 | 0.923 |
| **Chunk** (a chunk containing the answer) | 0.458 | 0.625 | 0.708 | 0.792 | 0.574 |

- English questions: source R@1 = 1.00; French questions on English papers: 0.75.
- The right document is almost always found; the right **passage** much less often. This is the first thing to improve (re-ranking).
- Mean top-1 similarity: 0.864 with an answer vs 0.803 without, too close for a score threshold to detect "I don't know".
- With 24 questions, one question ≈ 4 points: differences under ~5 points are noise.

Reaching this baseline required fixing the data first: Unicode NFKC normalization (PDF ligatures such as `ﬁ` broke words like "fields"), removing two PDFs whose text extracts as garbage, and removing a duplicated book.

## Progress

- [x] Python environment and GitHub repository
- [x] Cleaning of `.md` notes and text extraction from PDFs, with Unicode normalization (`cleaning.py`)
- [x] Chunking by headings with token-based size control (`chunking.py`)
- [x] Embeddings and in-memory semantic search (`embedder.py`)
- [x] Retrieval evaluation and baseline (`eval/`)
- [x] LanceDB storage: table, upsert, delete by source, cosine search, self-test (`vectors.py`)
- [ ] LanceDB search checked against the baseline (regression test in `eval/run_eval.py`)
- [ ] Re-ranking (cross-encoder), hybrid search (BM25)
- [ ] Incremental indexing
- [ ] Generation with an LLM
- [ ] API (FastAPI)
- [ ] Tests (pytest)
- [ ] Deployment on the homelab

## Repository structure

```
rag-obsidian/
├── cleaning.py         # cleaning of .md notes and PDF text extraction
├── chunking.py         # split cleaned notes into chunks for embedding
├── embedder.py         # embeddings with multilingual-e5-small + in-memory search
├── vectors.py          # LanceDB table: store chunk embeddings, delete by source, search
├── eval/
│   ├── questions.jsonl # evaluation questions, expected sources and evidence phrases
│   ├── run_eval.py     # recall@k and MRR at source and chunk level
│   └── results/        # one JSON file per evaluation run
├── requirements.txt    # pinned dependencies
├── .env.example        # expected variables, without values
├── .env                # local values and secrets (ignored by Git)
├── .gitignore
├── .gitattributes      # line endings normalised to LF
├── README.md
└── data/               # notes and PDFs (ignored by Git)
```

## Installation

Requirements: Anaconda (Windows), Git.

```
conda create -n rag-obsidian python=3.12
conda activate rag-obsidian
python -m pip install -r requirements.txt
```

Then copy `.env.example` to `.env` and fill in the values.

> [!warning] Environment rules
> - Install packages with `python -m pip`, never with `conda install` in this environment.
> - Run scripts with `python`, never with `py` (which ignores the conda environment).
> - In VS Code, select the interpreter `envs\rag-obsidian\python.exe`.

The first run of `embedder.py` downloads the model (~470 MB) into the Hugging Face cache (`~/.cache/huggingface/hub`); later runs reuse it.

### Environment variables

| Variable | Role |
|---|---|
| `STORAGE_DIR` | Folder containing the notes and PDFs to process |
| `CLEAN_DIR` | Folder for debugging outputs: cleaned notes, `chunks.jsonl`, embeddings cache |
| `DB_DIR` | Folder of the LanceDB database (outside the repository) |
| `HF_TOKEN` | Optional Hugging Face read token: faster and more reliable model download |

A missing path variable stops the script with an explicit error message. Only variable **names** are ever printed, never their values.

### Main dependencies

| Package | Use |
|---|---|
| `python-frontmatter` | Separate the YAML frontmatter (Obsidian properties) from the note body |
| `python-dotenv` | Read the `.env` file |
| `pymupdf` | Extract text from PDFs |
| `transformers` | Load the `multilingual-e5-small` tokenizer to count tokens |
| `sentence-transformers` | Load the embedding model and encode texts |
| `lancedb` (0.41.0) | Embedded vector database (installs `pyarrow`, used for the table schema) |
| `psutil` | Measure memory usage |
| `pytest` | Tests (installed, no tests in the repository yet) |

## Usage

```
python cleaning.py        # clean every note and write the result to CLEAN_DIR
python chunking.py        # clean + chunk every note, print size statistics, export chunks.jsonl
python embedder.py        # clean + chunk + embed, then run a test question and print the best chunks
python vectors.py         # self-test of the LanceDB layer (7 checks, cleans up after itself)
python -m eval.run_eval   # evaluate retrieval on eval/questions.jsonl (run from the repository root)
```

## How it works

### `cleaning.py`

| Element | Role |
|---|---|
| `cleanFile` (dataclass) | A cleaned document: `name`, `content`, `metadata` |
| `load_env_file()` | Loads `.env` with `python-dotenv` into `os.environ` |
| `env_path()` | Reads a path from an environment variable, with a clear error if it is missing |
| `normalize_unicode()` | NFKC normalization: PDF ligatures (`ﬁ`, `ﬂ`), full-width letters, non-breaking spaces become plain characters |
| `clean_wikilinks()` | Replaces `[[target]]`, `[[target#section]]`, `[[target\|alias]]` and `![[embed]]` with the target name |
| `clean_markdown()` | Normalizes, cleans wikilinks, separates the frontmatter, writes the cleaned note |
| `clean_pdf()` | Extracts and normalizes the text of every page with `pymupdf` |
| `clean_files()` | Processes every `.md` and `.pdf` in `STORAGE_DIR` |

### `chunking.py`

1. **Split by headings**: each Markdown heading starts a new section; a heading stack keeps the path (`Note > Section > Subsection`). `#` lines inside code blocks are ignored.
2. **Size control**: a section above `MAX_TOKENS` (300) is split by paragraphs, then lines, then sentences, then directly on tokens as a last resort. Small consecutive pieces are packed back together up to `MAX_TOKENS`.
3. **Merge small chunks**: chunks under `MIN_TOKENS` (50) are merged into the previous one when the result stays under `MAX_TOKENS`.

Each `Chunk` has `text`, `source`, `headings` and a `chunk_id` (SHA-256). `text_for_embedding` adds the `passage: ` prefix expected by E5 and the heading path. Tokens are counted with the embedding model's own tokenizer, because the model truncates anything above 512 tokens.

### `embedder.py`

| Function | Role |
|---|---|
| `get_model()` | Loads `SentenceTransformer(MODEL_NAME)` once and reuses it |
| `embed_passages(chunks)` | Encodes all chunks in one call (batches of 32), normalized vectors, shape `(n_chunks, 384)` |
| `embed_query(question)` | Adds the `query: ` prefix and encodes the question |
| `get_best_embedding(vectors, n, ref_vector)` | Scores = dot product (= cosine similarity, since vectors are normalized), returns the indices of the `n` best chunks |

### `vectors.py`

One row = one chunk. Columns: `id` (the `chunk_id`), `vector` (384 `float32`), `text` (the original text, without the `passage: ` prefix), `source`, `headings` (the heading path joined with ` > `). The database lives on disk in `DB_DIR`.

| Function | Role |
|---|---|
| `get_table()` | Opens the `chunks` table, or creates it (empty, explicit PyArrow schema) if it does not exist; the connection is opened once |
| `upsert_chunks(chunks, vectors)` | Writes the chunks with `merge_insert("id")`: updates the row if the id exists, inserts it otherwise, so writing twice never duplicates |
| `delete_by_source(source)` | Deletes every chunk of one source file (apostrophes in the name are escaped) |
| `search(query_vector, k)` | The `k` nearest chunks by cosine similarity, best first: dicts with `id`, `text`, `source`, `headings`, `score` (1.0 = identical) |

Running `python vectors.py` executes a self-test with assertions: insert, no duplicate on a second write, update of a modified chunk, search (a chunk is its own nearest neighbour), invalid arguments, delete with an apostrophe in the source name, cleanup. It only touches the fake sources `test.md` and `it's.md` and can be run repeatedly.

### `eval/run_eval.py`

- Each question in `questions.jsonl` has `expected_sources` (keys matched against file names) and `evidence` (short phrases from the passage that contains the answer).
- **Source level**: rank of the first chunk from an expected source. **Chunk level**: rank of the first chunk from an expected source that contains an evidence phrase (after normalization: lowercase, quotes, hyphenation, whitespace).
- Reports recall@1/3/5/10 and MRR overall, by question type and by language, plus the top-1 score of questions with vs without an answer.
- Flags questions whose evidence is found in **no** chunk (wrong evidence or missing source) and leaves them out of chunk-level metrics.
- Embeddings are cached in `CLEAN_DIR/embeddings_cache.npz` and recomputed only when chunk ids or the model change.

## Known limitations

- French questions are weaker than English ones (source R@1 0.75 vs 1.00); French course notes tend to match French questions by language rather than meaning.
- Small evaluation set (24 questions with an answer): small differences between runs are noise.
- Embedding time and RAM not measured yet.
- `chunk_id` includes the chunk position: adding a paragraph changes the ids of all following chunks (to fix before incremental indexing).
- `chunk_id` also depends on the text: a modified chunk gets a new id and the old row stays in the table, so incremental indexing must call `delete_by_source()` before `upsert_chunks()` for a modified note.
- LanceDB search not yet checked against the baseline; no real chunk is indexed yet.
- The `passage: ` prefix and heading path are not counted in `MAX_TOKENS`.
- PDFs whose fonts have no usable character mapping extract as garbage (would need OCR); PDF page numbers are lost (needed for citations).
- Cleaning: wikilinks are cleaned before the frontmatter is parsed, `[[note|alias]]` keeps the target instead of the alias, image embeds leave the file name in the text, the folder scan is not recursive.
- No automated tests yet (only the self-test of `vectors.py`).

## Next steps

1. Regression test: `eval/run_eval.py` uses LanceDB `search()`, with the same top-10 ids and the same baseline numbers as the in-memory search.
2. Incremental indexing (step 5): hash and registry, `delete_by_source()` then `upsert_chunks()` for each modified note.
3. Re-ranking of the top 20-30 chunks with a multilingual cross-encoder, measured against the baseline.
4. Hybrid search (BM25 + vectors, Reciprocal Rank Fusion).
5. LLM generation with an "I don't know" instruction, then question translation/rewriting for French queries.
6. Grow the evaluation set to 40-50 questions; PDF page numbers (`feature/pdf-pages`).
