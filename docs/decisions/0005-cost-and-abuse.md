# 0005 — Cost and abuse controls

## Context

The demo is public behind a shared password that will be printed on a resume, so it will leak. Every question costs Anthropic tokens, and Cloud Run bills while a connection is open.

## Decision

- **Measured cost:** $0.0051 per question on average, $0.0079 at most, across the 12 answerable cases of the e2e eval with the search tool (Haiku 4.5 at $1 / $5 per million input/output tokens, list price as of 2026-10-26). The earlier single-rewrite design measured $0.0016.
- **Bounded worst case per question:** at most 3 searches with research steps capped at 200 output tokens, at most 6 passages of ≤ ~260 tokens, 4 history turns, questions ≤ 500 characters, answers ≤ 700 tokens — about $0.02. Off-topic questions stop after the research step (about $0.001).
- **Hard stop:** a dedicated Anthropic workspace with a USD 20/month spend limit and a notification at 50%, so a leaked password can't affect other projects.
- **Per-tab cap:** 20 questions per browser session, counting every attempt and not reset by New chat. Reloading resets it, so it slows casual overuse; the spend limit is the hard stop.
- **Cloud Run:** at most one instance. A GCP budget of USD 70/month with alerts at 50% and 90%.
- **Errors:** two user messages ("Busy right now…" / "Demo unavailable right now.") from the API's error type, never a stack trace.

## Alternatives considered

| Alternative | Why not | What would change it |
| --- | --- | --- |
| Cloud Run IAM / IAP | Every reviewer would need a Google account sign-in. | Sharing only inside one organization. |
| No password | Anyone with the URL could spend the workspace limit. | A free-tier model or a much higher limit. |
| In-memory daily call budget | Resets on every cold start and redeploy, so it can't enforce a daily limit without a database. | Adding a persistent store for other reasons. |

## Consequences

- If the spend limit is reached, the demo shows "Demo unavailable right now." until the author rotates the password and raises or resets the limit.
- Every model call and every turn writes a JSON line to stdout, so Cloud Logging shows tokens, cost and search queries per session.
