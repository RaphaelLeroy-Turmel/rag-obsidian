---
tags: [projet/rag-obsidian, readme]
created: 2026-11-01
updated: 2026-11-19
status: in progress — step 2 (chunking)
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
  1. Cleaning                         (cleaning.py)  ✔
  2. Chunking (by Markdown headings)  (chunking.py)  ← current step
  3. Embeddings (multilingual-e5-small, CPU)
  4. Vector store (LanceDB)
  5. Incremental indexing (hash + SQLite registry)
        │
        ▼
  Question → passage retrieval → LLM via API → answer + sources
```

Main design choices: home-made pipeline (no LangChain/LlamaIndex, to understand each step), local embeddings so notes never leave the machine during indexing, embedded vector database (no extra service), LLM through an API because there is no RAM for a local one.

## Progress

- [x] Python environment and GitHub repository
- [x] Cleaning of `.md` notes and text extraction from PDFs (`cleaning.py`)
- [x] Chunking by headings with token-based size control (`chunking.py`)
- [ ] Chunking validated with the real tokenizer on my notes
- [ ] Tests (pytest)
- [ ] Embeddings
- [ ] In-memory search (first baseline)
- [ ] LanceDB storage + incremental indexing
- [ ] Generation with an LLM
- [ ] API (FastAPI) and evaluation (recall@k, MRR)
- [ ] Deployment on the homelab

## Repository structure

```
rag-obsidian/
├── cleaning.py         # cleaning of .md notes and PDF text extraction
├── chunking.py         # split cleaned notes into chunks for embedding
├── requirements.txt    # pinned dependencies
├── .env.example        # expected variables, without values
├── .env                # local values (ignored by Git)
├── .gitignore
├── .gitattributes      # line endings normalised to LF
├── README.md
└── data/               # test notes (ignored by Git)
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

### Environment variables

| Variable | Role |
|---|---|
| `STORAGE_DIR` | Folder containing the notes and PDFs to process |
| `CLEAN_DIR` | Folder where cleaned notes and exported chunks are written (debugging) |

A missing variable stops the script with an explicit error message.

### Main dependencies

| Package | Use |
|---|---|
| `python-frontmatter` | Separate the YAML frontmatter (Obsidian properties) from the note body |
| `python-dotenv` | Read the `.env` file |
| `pymupdf` | Extract text from PDFs |
| `transformers` | Load the `multilingual-e5-small` tokenizer to count tokens |
| `pytest` | Tests (installed, no tests in the repository yet) |

## Usage

```
python cleaning.py     # clean every note and write the result to CLEAN_DIR
python chunking.py     # clean + chunk every note, print size statistics and a preview
```

`chunking.py` prints the number of chunks and their min / median / max size in tokens, shows the first 10 chunks, and exports all chunks to `CLEAN_DIR/chunks.jsonl` for manual inspection.

## How it works

### `cleaning.py`

| Element | Role |
|---|---|
| `cleanFile` (dataclass) | A cleaned document: `name`, `content`, `metadata` |
| `load_env_file()` | Loads `.env` with `python-dotenv` into `os.environ` |
| `env_path()` | Reads a path from an environment variable, with a clear error if it is missing |
| `clean_wikilinks()` | Replaces `[[target]]`, `[[target#section]]`, `[[target\|alias]]` and `![[embed]]` with the target name |
| `clean_markdown()` | Cleans wikilinks, separates the frontmatter, writes the cleaned note |
| `clean_pdf()` | Extracts the text of every page with `pymupdf` |
| `clean_files()` | Processes every `.md` and `.pdf` in `STORAGE_DIR` |

### `chunking.py`

1. **Split by headings**: each Markdown heading starts a new section; a heading stack keeps the path (`Note > Section > Subsection`). `#` lines inside code blocks are ignored.
2. **Size control**: a section above `MAX_TOKENS` (300) is split by paragraphs, then lines, then sentences, then directly on tokens as a last resort. Small consecutive pieces are packed back together up to `MAX_TOKENS`.
3. **Merge small chunks**: chunks under `MIN_TOKENS` (50) are merged into the previous one when the result stays under `MAX_TOKENS`.

Each `Chunk` has `text`, `source`, `headings` and a `chunk_id` (SHA-256). `text_for_embedding` adds the `passage: ` prefix expected by E5 and the heading path. Tokens are counted with the embedding model's own tokenizer, because the model truncates anything above 512 tokens.

## Known limitations

- `chunk_id` includes the chunk position: adding a paragraph changes the ids of all following chunks (to fix before incremental indexing).
- The `passage: ` prefix and heading path are not counted in `MAX_TOKENS`.
- PDF page numbers are lost when pages are joined (needed for citations).
- Cleaning: wikilinks are cleaned before the frontmatter is parsed, `[[note|alias]]` keeps the target instead of the alias, image embeds leave the file name in the text, the folder scan is not recursive.
- Not handled yet: `%%…%%` comments, callouts, inline tags, Markdown links, Unicode normalisation, PDF headers/footers and hyphenation.
- No automated tests yet.

## Next steps

1. Validate chunking with the real tokenizer on my notes.
2. Keep PDF page numbers.
3. Write `pytest` tests for cleaning and chunking.
4. Step 3: embeddings (`embed_passages()`, `embed_query()`).
