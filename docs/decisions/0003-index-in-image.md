# 0003 — Search index built into the image; no uploads

## Context

The earlier app let anyone upload a PDF, and re-indexed silently when its collection was empty. A public demo can't offer uploads (anyone could add or overwrite documents and run up costs), and silent re-indexing would hide a broken deployment.

## Decision

- `healthrag.build_index` runs during `docker build`. It loads the PDFs listed in `corpus.toml`, skips cover, contents and reference pages, chunks, embeds, writes the Chroma index, runs a warm-up query, checks that every source has at least 20 chunks, and writes `index_manifest.json` (labels, editions, URLs, chunk counts, embedding model, query prefix).
- The runtime never imports the build code (enforced by `tests/test_boundaries.py`). It opens the existing collection without creating one and reads labels and editions from the manifest.
- Documents change only by editing `corpus.toml` and rebuilding the image.

## Alternatives considered

| Alternative | Why not | What would change it |
| --- | --- | --- |
| Local admin upload mode | Two ways to add documents, and one wrong flag would expose upload publicly. | A version with real authentication and per-user indexes. |
| Build the index at container start | Slower cold starts, and a model download at runtime. | A corpus too large to fit in the image. |

## Consequences

- Rolling back to an earlier Cloud Run revision also rolls back the index.
- The container needs no network access at runtime except the Anthropic API (`HF_HUB_OFFLINE=1`).
