# 0004 — Search tool, relevance cutoff and history policy

## Context

The first version made one Claude call to rewrite follow-ups into a standalone query, searched once, and always passed the top 3 passages to the answer call. Two problems showed up in use:

- A follow-up like "What about kids?" needs the conversation to make sense. Searching with the raw follow-up finds the right passage for only 1 of 5 follow-up cases.
- Search always returns 3 passages, even when none is about the question. Borderline or off-topic questions got answers built on, and source chips pointing to, loosely related pages.

## Decision

**Research step: a bounded search-tool loop.**

- Claude Haiku gets one tool, `search_guides(query, guide)`, plus a description of what each guide covers. It sees the last 2 turns (answers cut to 300 characters) and the new question.
- It writes standalone queries, can search again with other words or in the other guide, and replies `DONE` when it has enough. Each step is capped at 200 output tokens, and the whole loop at 3 searches.
- The visitor's source filter, if set, overrides the guide the model picks.
- If the model doesn't search, or a research call fails, the app searches once with the original question.

**Relevance cutoff.**

- Passages with a cosine distance above a per-model cutoff are dropped before the model or the visitor sees them: 0.45 for bge-small, 0.78 for MiniLM.
- The cutoff sits between the measured on-topic maximum (0.39) and off-topic minimum (0.53).
- If no passage survives, the app replies that the guides don't cover the question, lists what they do cover, and makes no answer call.

**Answer step.**

- A separate streamed call gets the last 4 turns as plain question/answer messages (no earlier excerpts), up to 6 passages found in this turn (deduplicated, closest first), and the **original** question.
- Facts must come from those passages and are cited as (Guide, page). The UI shows chips only for pages the answer actually cites. The e2e eval checks every citation against the retrieved pages.

## Alternatives considered

| Alternative | Why not | What would change it |
| --- | --- | --- |
| Single rewrite call (v1) | Cheaper ($0.0016 vs $0.0051 per question) and just as accurate on this eval set, but it can't search twice or switch guides, and it had no way to say "not covered". | Cost mattering more than handling borderline questions. |
| Join the previous and current question (no extra call) | Free, and 5/5 here. Kept as a measured baseline. It drifts when the topic changes and lengthens queries. | A larger follow-up set showing no gain from model-written queries. |
| Let the model answer from its own knowledge when the guides don't cover a question | Removes the guarantee that every fact is sourced, which matters for health information. | A clearly labeled "general knowledge" mode that the visitor turns on. |
| Let the research step write the final answer | Saves a call, but the answer couldn't stream, and the grounding rules would be mixed with tool-use instructions. | Streaming tool-use responses becoming simple to handle in the UI. |
| Re-send earlier excerpts | Adds tokens every turn, and citations become harder to trace to the current search. | Follow-ups needing to quote earlier passages. |

## Consequences

- Cost per question roughly triples, to about half a cent. The worst case is about two cents (3 searches, 6 passages, 4 turns of history).
- Every search is shown under the answer and logged, so a query that changed the visitor's meaning is visible.
- The cutoff is calibrated on 14 cases. It should be re-checked whenever the corpus or the embedding model changes. The retrieval eval in CI checks that off-topic queries are still rejected and no expected passage is dropped.
