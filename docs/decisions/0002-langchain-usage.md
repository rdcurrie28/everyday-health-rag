# 0002 — LangChain building blocks, FastEmbed with bge-small, no LangGraph

## Context

The pipeline needs page-aware PDF loading, size-capped chunking, embeddings, a persistent vector store with a source filter, and Claude calls whose token usage is logged. The target roles list LangChain experience.

## Decision

| Job | Piece |
| --- | --- |
| PDF loading | `PyPDFLoader` (one document per page, with `page` and `page_label`) |
| Chunking | `RecursiveCharacterTextSplitter` (1,000 characters, 150 overlap, per page) |
| Embeddings | `FastEmbedEmbeddings` with `BAAI/bge-small-en-v1.5` (ONNX, explicit cache folder) |
| Vector store | `langchain-chroma` over a persistent Chroma collection (cosine) |
| Model calls | `ChatAnthropic`, so the conversation code isn't tied to one provider's message format |

The conversation flow (search-tool loop → answer) is plain Python. The tool loop uses `ChatAnthropic.bind_tools` with one tool and is bounded at 3 searches; see [0004](0004-rewrite-and-history.md). It doesn't use `ConversationalRetrievalChain`, `create_history_aware_retriever`, `create_agent` or a LangGraph graph. The loop has one tool, a fixed exit, and no state that outlives a question, and an explicit loop keeps the search limit, the fallback, the relevance cutoff and per-call cost logging in plain view and testable with a fake model.

FastEmbed does not add bge's query instruction, so `retrieval.py` adds `"Represent this sentence for searching relevant passages: "` to queries for bge models only. The prefix is recorded in the index manifest.

## Alternatives considered

| Alternative | Why not | What would change it |
| --- | --- | --- |
| Hand-rolled pipeline | Page-aware loading, splitting and a provider-neutral model interface would all be hand-written. | LangChain upgrades breaking the build repeatedly. |
| LangGraph, `create_agent`, or legacy chains | They hide the parts this project measures. The one tool loop here is about 40 lines; a graph would add a framework layer without adding behavior. | Several tools, branching between different steps, human approval steps, or conversation state saved across sessions (checkpointing). |
| `anthropic` SDK directly | Ties the conversation code to one provider. | The wrapper lagging behind a needed API feature. |
| Chroma's built-in all-MiniLM-L6-v2 | Reads only 256 tokens (the 1,000-character chunks reach ~260). Measured as a comparison index: same hit rate on this eval set. | MiniLM beating bge on a larger eval set. |
| `langchain-huggingface` | Pulls in PyTorch (~1–2 GB). | Needing a model FastEmbed doesn't support. |

## Consequences

- `llm.py` still imports the `anthropic` SDK, because errors raised mid-stream have to be classified from the SDK's error payload.
- `langchain-community` (used for `PyPDFLoader` and `FastEmbedEmbeddings`) now warns that it is being sunset. It works today and versions are locked; moving those two classes to standalone packages is a planned follow-up.
