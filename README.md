# Everyday Health Chat

Ask follow-up questions about the US *Dietary Guidelines for Americans* and *Physical Activity Guidelines for Americans*, and get short answers that cite the guide and page.

> Demo GIF: _to be recorded after the first deploy._

**Live demo:** <https://health-rag-app-1054101628971.us-central1.run.app>. The password is on the author's resume or available on request (open an issue).

## What it shows

- **Agentic retrieval-augmented generation (RAG).** Claude Haiku gets a `search_guides` tool. It writes its own standalone queries (so "What about kids?" works as a follow-up), can search again with other words or in the other guide, and stops when it has enough, with at most 3 searches per question. The UI shows every search under the answer.
- **Honest "not covered" answers.** Passages less relevant than a measured cutoff are dropped. If nothing relevant is left, the app says the guides don't cover the question instead of showing loosely related pages, and skips the answer call.
- **Grounded, checked citations.** The final answer is a separate streamed call that may use only the passages found. Every fact is cited as (Guide, page), the source chips show only pages the answer actually cites, and the end-to-end eval checks every citation against the retrieved pages.
- **Measured retrieval quality.** A 14-case eval compares query strategies, three indexes, and two conversation designs.
- **Known, bounded cost.** Every model call logs tokens and cost. A question averages $0.005.
- **A reproducible build.** The embedding model and the search index are built into the Docker image, so the container makes no network calls at runtime except to Anthropic.

## Why these two guides

Both are US federal publications (public domain), everyone can relate to them, and they're full of figures that change by age group, which makes follow-ups natural. The corpus is the newest *Dietary Guidelines* (2025–2030, released January 2026; 10 pages) and the *Physical Activity Guidelines*, 2nd edition (2018; 118 pages).

## Architecture

```text
BUILD TIME (docker build)
data/*.pdf ─► PyPDFLoader (per page, printed page labels)
           ─► skip cover/contents/references (corpus.toml)
           ─► RecursiveCharacterTextSplitter (1,000 chars, 150 overlap, "Guide · p. N" prefix)
           ─► FastEmbed bge-small-en-v1.5 (ONNX) ─► Chroma (cosine) ─► index_manifest.json

RUN TIME (Cloud Run, 1 instance)
Browser ─► app.py (Streamlit: password gate, session cap, chat)
             └─ conversation.research  (bounded tool loop, ≤3 searches)
                  ├─ Claude Haiku + search_guides tool (≤200 tokens per step)
                  └─ search_guides ─► Chroma (k = 3, optional guide) ─► drop passages past the cutoff
             └─ nothing relevant? ─► "not covered" reply, no answer call
             └─ conversation.answer_stream ─► Claude Haiku (≤700 tokens, streamed, ≤6 passages)
             └─ logs.py ─► one JSON line per call and per turn ─► Cloud Logging
```

| Module | Job |
| --- | --- |
| `healthrag/config.py` | Constants and the few environment settings (mode, paths, secrets, git SHA) |
| `healthrag/corpus.py`, `build_index.py` | Build time only: read `corpus.toml`, chunk, embed, write the index and manifest |
| `healthrag/retrieval.py` | Open the existing index (never create one), return `Chunk` objects, apply the per-model relevance cutoff |
| `healthrag/llm.py` | The only place Claude is called; every call is logged with tokens and cost, including interrupted streams |
| `healthrag/conversation.py` | Search-tool loop, prompts, "not covered" path, citation check (no Streamlit) |
| `app.py` | The Streamlit UI |

## Evaluation

14 hand-written cases (`evals/questions.jsonl`): 7 first-turn questions, 5 follow-ups (age group, amount, and topic changes), and 2 off-topic questions. A case is a hit if an expected phrase from the guide appears in a retrieved passage.

**Retrieval only (follow-up hits · overall hit@3), 2026-10-26.** No model calls; runs in CI.

| Index | Raw follow-up | Joined (previous + current question) | Gold query (upper bound) |
| --- | --- | --- | --- |
| Before fixes (4,000-char chunks, no page skipping, no prefix) | 1/5 · 0.67 | 4/5 · 0.92 | 5/5 · 1.00 |
| **Tuned (bge-small, 1,000-char chunks)** | 1/5 · 0.67 | 5/5 · 1.00 | 5/5 · 1.00 |
| MiniLM comparison (same chunks) | 1/5 · 0.67 | 5/5 · 1.00 | 5/5 · 1.00 |

All indexes hit 7/7 first-turn questions. The relevance cutoff (distance ≤ 0.45 for bge-small, 0.78 for MiniLM) removes no expected passage and rejects every off-topic query.

**End to end (tuned index, live Claude), two designs compared:**

| | Single rewrite call (v1) | Search tool (current) |
| --- | --- | --- |
| Follow-up hits | 5/5 | 5/5 |
| First-turn hits | 6/6 | 7/7 |
| Off-topic answered "not covered" | not measured (no cutoff) | 2/2 |
| Answers whose citations all match retrieved pages | 11/11 | 12/12 |
| Searches per question | 1 | 1.08 |
| Mean / max cost per question | $0.0016 / $0.0022 | $0.0051 / $0.0079 |
| Mean time to first answer token | 516 ms | 407 ms |

What the numbers do and don't show:

- Turning a follow-up into a standalone query is what matters most: raw follow-ups hit 1 of 5. On this small set, simply joining the previous and current question also hits 5 of 5.
- The search tool doesn't raise the hit rate here. It adds the ability to search again or in the other guide (used on borderline questions such as "Is coffee okay to drink?"), and it costs about 3× more per question. A larger, harder eval set is needed to show its value in numbers.
- The three indexes score the same hit rate. The tuned index wins on other grounds: chunks fit the embedding model (≈260 tokens vs ≈970 in the baseline, which bge truncates at 512), page citations are exact, and answers carry fewer excerpt tokens.

Results files are in `evals/results/`.

## Cost

| | Per question |
| --- | --- |
| Measured average (e2e eval) | $0.0051 |
| Measured maximum | $0.0079 |
| Worst case by construction (3 searches, 6 passages, 4 history turns) | ≈ $0.02 |
| Off-topic question ("not covered", no answer call) | ≈ $0.001 |

Haiku 4.5 at $1 / $5 per million input/output tokens (list price as of 2026-10-26). Cloud Run scales to zero when idle. One always-on instance (1 vCPU, 2 GiB; the `--warm` setting, or a forgotten open tab) costs roughly $40–65 a month at list prices.

Cold start: _measured after the first deploy._

## Risks and known limits

| Risk or limit | Mitigation |
| --- | --- |
| A leaked password uses up the spend limit | USD 20/month workspace limit with a 50% notification; per-tab cap; password rotation |
| A forgotten open tab keeps the instance billed | One instance at most; GCP budget alert at USD 70/month |
| The adult key guideline (PDF pp. 8, 56) is outranked by the older-adults chapter, which repeats the same 150–300 minute figure | Answers stay correct but may cite the older-adults page; tracked by the eval |
| The eval is tuned on the same 14 cases it scores | Stated here; the example questions are taken from passing cases |
| A search query changes what the visitor meant | Every search is shown under the answer; if the model doesn't search, the original question is searched |
| A borderline question gets loosely related passages | Relevance cutoff; the answer says what the guides don't cover; chips show only cited pages |
| Numbers that appear only in charts can't be retrieved | Known limit of text extraction |
| Chat history is lost on reload, redeploy or scale-to-zero | Stated in the app footer |
| `langchain-community` is being sunset | Versions are locked; move `PyPDFLoader` and `FastEmbedEmbeddings` to standalone packages |
| Not medical advice | System prompt rule and footer disclaimer |

## Decision records

1. [UI and hosting](docs/decisions/0001-ui-and-hosting.md)
2. [LangChain usage, embeddings, no LangGraph](docs/decisions/0002-langchain-usage.md)
3. [Index built into the image, no uploads](docs/decisions/0003-index-in-image.md)
4. [Search tool, relevance cutoff and history policy](docs/decisions/0004-rewrite-and-history.md)
5. [Cost and abuse controls](docs/decisions/0005-cost-and-abuse.md)

## Run locally

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env          # set ANTHROPIC_API_KEY; leave APP_PASSWORD empty to skip the gate
uv run python -m healthrag.build_index      # ~1–2 minutes; downloads the embedding model once
uv run streamlit run app.py
```

Tests and evals:

```bash
uv run pytest --cov=healthrag.conversation --cov=healthrag.llm --cov-branch --cov-fail-under=100
uv run python -m evals.run_eval --mode retrieval --label mytest --no-save
uv run python -m evals.run_eval --mode e2e --label mytest     # calls Claude, ~7 cents
```

Comparison indexes:

```bash
uv run python -m healthrag.build_index --baseline --index-dir .index-baseline
uv run python -m healthrag.build_index --embedding-model sentence-transformers/all-MiniLM-L6-v2 --index-dir .index-minilm
uv run python -m evals.run_eval --mode retrieval --label baseline --index-dir .index-baseline --no-gate
```

## Deploy

See the one-time setup at the top of [`deploy.sh`](deploy.sh): APIs, an Artifact Registry repo, two secrets, a runtime service account, the GCP budget and the Anthropic workspace limit. Then:

```bash
PROJECT_ID=my-project REGION=us-central1 AR_REPO=apps SERVICE=health-rag-app ./deploy.sh          # scale to zero
PROJECT_ID=my-project REGION=us-central1 AR_REPO=apps SERVICE=health-rag-app ./deploy.sh --warm   # keep one instance
```

Post-deploy smoke test:

1. A wrong password is refused.
2. "How many minutes of physical activity do adults need each week?" is answered with a page citation.
3. "What about kids?" shows "Searched for: …" and cites the children's guideline.
4. "What is the capital of France?" gets the "not covered" reply with no citations.
5. The source filter in ⚙ limits results to one guide.
6. **New chat** clears the history but not the question count.
7. Cloud Logging query `jsonPayload.event="llm_call"` shows tokens and cost.
8. Question 21 in one tab is refused.

## License

Code: MIT. The two guides are US government works and are in the public domain.
