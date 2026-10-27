# 0001 — Streamlit chat UI on Google Cloud Run

## Context

The project replaces a default Streamlit search page (sidebar of controls, one question box) with a chat app that hiring reviewers can open from a link. It has one author, no team, and a small UI: a chat, a settings menu, and source citations.

## Decision

Streamlit's chat components (`st.chat_message`, `st.chat_input`, `st.write_stream`) with a custom theme in `.streamlit/config.toml` and under 60 lines of CSS. Deployed as a single container on Cloud Run.

## Alternatives considered

| Alternative | Why not | What would change it |
| --- | --- | --- |
| Reflex | Node, a build step and two servers for a one-screen app; long Claude calls need async event handlers. | Needing layouts or interactions Streamlit can't produce. |
| PatternFly (React) + FastAPI | Two languages and toolchains, an estimated 400–600 lines of glue code, and a default theme that doesn't match the palette. | A team with React developers, or several clients sharing one API. |
| Streamlit Community Cloud / HF Spaces | Free and simple, but shows no container or cloud skills, and the index has to be rebuilt on a temporary disk. | The project no longer needs to show cloud skills. |

## Consequences

- Streamlit holds a WebSocket per tab and keeps chat history in process memory, so Cloud Run runs at most one instance with session affinity and a one-hour request timeout.
- A forgotten open tab keeps the instance active. With one instance, that costs the same as the `--warm` setting; the README prices it.
