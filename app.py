from html import escape
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import streamlit as st

from healthrag.config import (
    MAX_QUESTION_CHARS,
    MAX_TURNS_PER_SESSION,
    check_password,
    get_settings,
)
from healthrag.conversation import (
    GUIDE_SCOPE,
    NOT_COVERED,
    Research,
    Turn,
    answer_stream,
    check_citations,
    finish_turn,
    log_turn,
    prepare_turn,
)
from healthrag.llm import LLMError
from healthrag.logs import log_event
from healthrag.retrieval import Chunk, IndexNotReady, open_retriever

EXAMPLE_GROUPS = {
    "🏃 Physical activity": [
        "How many minutes of physical activity do adults need each week?",
        "How active should preschool-aged children be?",
        "Is it safe to exercise during pregnancy?",
    ],
    "🥦 Nutrition": [
        "How much sodium should people eat per day?",
        "What should infants be fed in the first 6 months?",
        "How many servings of dairy should I have per day?",
    ],
}
EXAMPLES = [question for group in EXAMPLE_GROUPS.values() for question in group]
FOLLOW_UP = "What about kids?"
ERROR_MESSAGES = {
    "busy": "Busy right now. Please try again in a minute.",
    "unavailable": "Demo unavailable right now.",
}
FOOTER = (
    "Not medical advice. Answers are generated from excerpts of the two guides and may contain "
    "errors. Chat history clears when the page reloads."
)
ALL_SOURCES = "Both guides"

st.set_page_config(page_title="Everyday Health Chat", page_icon="🥗", layout="centered")
st.html(Path(__file__).parent / ".streamlit" / "style.css")

settings = get_settings()
if settings.deployed and settings.missing_secrets():
    st.error("Demo not configured.")
    st.stop()


@st.cache_resource(show_spinner="Loading the search index…")
def load_retriever():
    retriever = open_retriever(settings)
    log_event(
        "startup",
        git_sha=settings.git_sha,
        embedding_model=retriever.manifest["embedding_model"],
        chunks={sid: info["chunks"] for sid, info in retriever.manifest["sources"].items()},
        streamlit=version("streamlit"),
        langchain_core=version("langchain-core"),
    )
    return retriever


try:
    retriever = load_retriever()
except IndexNotReady as e:
    st.error(str(e))
    st.stop()

state = st.session_state
state.setdefault("session_id", uuid4().hex[:8])
state.setdefault("turns", [])
state.setdefault("turn_count", 0)
state.setdefault("session_cost", 0.0)
state.setdefault("authed", False)

if settings.app_password and not state.authed:
    st.title("Everyday Health Chat")
    st.write(
        "Ask questions about the Dietary Guidelines for Americans and the Physical Activity "
        "Guidelines for Americans, including follow-ups. Answers cite the guide and page."
    )
    st.caption(
        "The password is on the author's resume or available on request through the GitHub "
        "README. Search queries are logged for 30 days to debug retrieval."
    )
    with st.form("login"):
        entered = st.text_input("Password", type="password")
        if st.form_submit_button("Enter", type="primary"):
            if check_password(entered, settings.app_password):
                state.authed = True
                st.rerun()
            st.error("Wrong password.")
    st.stop()

sources = retriever.manifest["sources"]
source_options = {ALL_SOURCES: None} | {info["label"]: sid for sid, info in sources.items()}

title_col, settings_col = st.columns([8, 1], vertical_alignment="bottom")
with title_col:
    st.title("Everyday Health Chat")
    st.caption(
        "Answers come only from "
        + " and ".join(f"*{info['label']}* ({info['edition']})" for info in sources.values())
        + "."
    )
with settings_col:
    with st.popover("⚙", help="Settings"):
        st.radio("Search in", list(source_options), key="source_choice")
        st.caption(f"Session cost: ${state.session_cost:.4f}")
        if st.button("New chat", width="stretch"):
            state.turns = []
            st.rerun()

with st.expander("What can I ask about?", expanded=not state.turns):
    for sid, info in sources.items():
        scope = GUIDE_SCOPE.get(sid)
        if scope:
            st.markdown(f"**{info['label']}** — {scope.split(': ', 1)[-1]}")
    st.caption('Questions outside these topics get a short "not covered" reply instead of a guess.')

source_filter = source_options[state.get("source_choice", ALL_SOURCES)]


def render_searches(research: Research | None) -> None:
    if research is not None and research.queries:
        st.caption("Searched for: " + " · ".join(f"“{query}”" for query in research.queries))


def render_cited(answer: str, chunks: list[Chunk]) -> None:
    retrieved = {(chunk.label, chunk.page_ref) for chunk in chunks}
    _, citations = check_citations(answer, chunks)
    cited = [c for c in dict.fromkeys(citations) if c in retrieved]
    if cited:
        st.html(
            '<div class="source-chips"><span class="chips-label">Cited:</span>'
            + "".join(
                f'<span class="source-chip">{escape(label)} · {escape(page)}</span>'
                for label, page in cited
            )
            + "</div>"
        )


def render_turn(turn: Turn) -> None:
    with st.chat_message("user"):
        st.markdown(turn.question)
    with st.chat_message("assistant"):
        render_searches(turn.research)
        st.markdown(turn.answer)
        render_cited(turn.answer, turn.chunks)
        if turn.chunks:
            with st.expander(f"Passages retrieved ({len(turn.chunks)})"):
                for chunk in turn.chunks:
                    st.markdown(
                        f"**{chunk.label} · {chunk.page_ref}** · distance {chunk.distance:.3f}"
                    )
                    st.text(chunk.text)
        st.caption(f"Cost: ${turn.cost_usd:.4f}")


def ask(question: str) -> None:
    if state.turn_count >= MAX_TURNS_PER_SESSION:
        st.warning(
            f"Session limit reached ({MAX_TURNS_PER_SESSION} questions). "
            "Reload the page to start a new chat."
        )
        log_event("cap_hit", session=state.session_id, turn_count=state.turn_count)
        return
    state.turn_count += 1
    history: list[Turn] = state.turns
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the guides…"):
                prepared = prepare_turn(
                    question,
                    history,
                    source_filter,
                    retriever,
                    session=state.session_id,
                    turn_no=state.turn_count,
                )
        except ValueError as e:
            st.warning(str(e))
            return
        state.session_cost += prepared.research.cost_usd
        render_searches(prepared.research)
        if not prepared.covered:
            st.markdown(NOT_COVERED)
            log_turn(prepared, "not_covered")
            state.turns.append(finish_turn(prepared, None))
            st.rerun()
        result = answer_stream(prepared, history)
        placeholder = st.empty()
        outcome = "answered"
        try:
            with placeholder.container():
                st.write_stream(result)
        except LLMError as e:
            outcome = f"error_{e.kind}"
            placeholder.error(ERROR_MESSAGES[e.kind])
        finally:
            result.close()
            if result.record.aborted:
                outcome = "aborted"
            state.session_cost += result.record.cost_usd or 0.0
            log_turn(prepared, outcome)
    if outcome == "answered":
        state.turns.append(finish_turn(prepared, result))
        st.rerun()


for turn in state.turns:
    render_turn(turn)

if not state.turns:
    st.markdown("**Try one of these:**")
    columns = st.columns(len(EXAMPLE_GROUPS))
    for column, (group, questions) in zip(columns, EXAMPLE_GROUPS.items(), strict=True):
        column.markdown(f"**{group}**")
        for question in questions:
            if column.button(question, key=f"example-{EXAMPLES.index(question)}", width="stretch"):
                state.pending_question = question
elif len(state.turns) == 1 and state.turns[0].question == EXAMPLES[0]:
    if st.button(FOLLOW_UP, key="follow-up"):
        state.pending_question = FOLLOW_UP

if source_filter:
    st.caption(f"Searching only: {state.source_choice}")

typed = st.chat_input("Ask about the guides", max_chars=MAX_QUESTION_CHARS)
question = typed or state.pop("pending_question", None)
if question:
    ask(question)

st.caption(FOOTER)
