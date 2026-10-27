import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from healthrag import llm
from healthrag.config import ANSWER_HISTORY_TURNS, MAX_QUESTION_CHARS, MAX_SEARCHES
from healthrag.conversation import (
    NO_RESULTS,
    NOT_COVERED,
    SEARCH_TOOL,
    Turn,
    answer_stream,
    build_answer_messages,
    check_citations,
    finish_turn,
    log_turn,
    prepare_turn,
    research,
    research_messages,
)
from healthrag.llm import LLMError

from .conftest import FakeRetriever, done_reply, make_chunk, tool_reply


def turns(n: int) -> list[Turn]:
    return [Turn(question=f"q{i}", source_filter=None, answer=f"a{i}") for i in range(n)]


def run(fake_chat, retriever=None, history=None, source_filter=None, question="What about kids?"):
    return research(
        question,
        history or [],
        source_filter,
        retriever or FakeRetriever(),
        session="s",
        turn_no=1,
    )


def test_research_runs_tool_searches_until_done(fake_chat):
    fake_chat.responses = [tool_reply("kids activity", guide="pag"), done_reply()]
    retriever = FakeRetriever()
    found, chunks = run(fake_chat, retriever)
    assert found.queries == ["kids activity"]
    assert retriever.calls == [("kids activity", "pag")]
    assert not found.fallback
    assert len(found.records) == 2
    assert chunks == retriever.chunks
    assert fake_chat.bound_tools == [[SEARCH_TOOL], [SEARCH_TOOL]]
    tool_messages = [m for m in fake_chat.invocations[1][0] if isinstance(m, ToolMessage)]
    assert tool_messages[0].tool_call_id == "toolu_0"
    assert '<excerpt source="Mini Guide" page="PDF p. 1">' in tool_messages[0].content


def test_user_source_filter_overrides_model_choice(fake_chat):
    fake_chat.responses = [tool_reply("sodium", guide="pag"), done_reply()]
    retriever = FakeRetriever()
    run(fake_chat, retriever, source_filter="dga")
    assert retriever.calls == [("sodium", "dga")]


def test_unknown_guide_searches_both(fake_chat):
    fake_chat.responses = [tool_reply("sodium", guide="both"), done_reply()]
    retriever = FakeRetriever()
    run(fake_chat, retriever)
    assert retriever.calls == [("sodium", None)]


def test_search_limit_is_enforced(fake_chat):
    fake_chat.responses = [tool_reply("a", "b"), tool_reply("c", "d"), tool_reply("e")]
    retriever = FakeRetriever()
    found, _ = run(fake_chat, retriever)
    assert found.queries == ["a", "b", "c"][:MAX_SEARCHES]
    assert len(fake_chat.invocations) == 2
    last_tools = [m for m in fake_chat.invocations[-1][0] if isinstance(m, ToolMessage)]
    assert all(m.tool_call_id for m in last_tools)


def test_limit_reached_message_for_extra_calls(fake_chat):
    fake_chat.responses = [tool_reply("a", "b", "c", "d"), done_reply()]
    found, _ = run(fake_chat)
    assert len(found.queries) == MAX_SEARCHES
    assert len(fake_chat.invocations) == 1


def test_no_search_falls_back_to_question(fake_chat):
    fake_chat.responses = [done_reply("Here is my answer without searching.")]
    retriever = FakeRetriever()
    found, chunks = run(fake_chat, retriever)
    assert found.fallback and found.queries == ["What about kids?"]
    assert chunks


def test_llm_error_falls_back_and_keeps_record(fake_chat):
    fake_chat.invoke_error = RuntimeError("boom")
    found, _ = run(fake_chat)
    assert found.fallback and found.queries == ["What about kids?"]
    assert found.records[0].error == "RuntimeError"


def test_llm_error_without_record_still_falls_back(monkeypatch):
    def fail(*args, **kwargs):
        raise LLMError("busy", "overloaded_error")

    monkeypatch.setattr(llm, "invoke", fail)
    found, chunks = run(None)
    assert found.fallback and found.records == [] and chunks


def test_irrelevant_results_are_dropped(fake_chat):
    far = make_chunk(1, text="far")
    far = type(far)(**{**far.__dict__, "distance": 0.9})
    retriever = FakeRetriever(chunks=[far])
    fake_chat.responses = [tool_reply("capital of France"), done_reply()]
    found, chunks = run(fake_chat, retriever)
    assert chunks == []
    tool_message = [m for m in fake_chat.invocations[1][0] if isinstance(m, ToolMessage)][0]
    assert tool_message.content == NO_RESULTS


def test_chunks_are_deduplicated_and_capped(fake_chat):
    chunks = [make_chunk(n) for n in range(5)]
    retriever = FakeRetriever(chunks=chunks)
    fake_chat.responses = [tool_reply("a", "b"), done_reply()]
    _, found = run(fake_chat, retriever)
    assert [c.id for c in found] == [c.id for c in chunks]


def test_blank_tool_query_is_ignored(fake_chat):
    fake_chat.responses = [tool_reply("   "), done_reply()]
    retriever = FakeRetriever()
    found, _ = run(fake_chat, retriever)
    assert found.fallback and retriever.calls == [("What about kids?", None)]


def test_research_messages_trim_history():
    history = turns(5)
    history[-1].answer = "x" * 1000
    messages = research_messages("And kids?", history)
    assert isinstance(messages[0], SystemMessage)
    assert [m.content for m in messages[1:3]] == ["q3", "a3"]
    assert len(messages[4].content) == 300
    assert messages[-1].content == "And kids?"


def test_answer_messages_trim_history_and_exclude_old_excerpts():
    history = turns(6)
    history[0].chunks = [make_chunk(text="OLD EXCERPT")]
    chunks = [make_chunk(0), make_chunk(1, label="Dietary Guidelines", page_ref="p. 5")]
    messages = build_answer_messages("And kids?", history, chunks)
    assert isinstance(messages[0], SystemMessage)
    assert len(messages) == 1 + 2 * ANSWER_HISTORY_TURNS + 1
    assert isinstance(messages[1], HumanMessage) and isinstance(messages[2], AIMessage)
    assert messages[1].content == "q2"
    final = messages[-1].content
    assert "OLD EXCERPT" not in "".join(str(m.content) for m in messages)
    assert '<excerpt source="Dietary Guidelines" page="p. 5">' in final
    assert final.endswith("Question: And kids?")


def test_no_excerpts_message():
    assert "No excerpts" in build_answer_messages("q", [], [])[-1].content


@pytest.mark.parametrize("question", ["", "   ", "x" * (MAX_QUESTION_CHARS + 1)])
def test_prepare_turn_rejects_bad_questions(question):
    with pytest.raises(ValueError):
        prepare_turn(question, [], None, FakeRetriever(), session="s", turn_no=1)


def test_full_turn_uses_original_question(fake_chat, log_lines):
    history = turns(1)
    prepared = prepare_turn(
        "What about kids?", history, None, FakeRetriever(), session="s", turn_no=2
    )
    assert prepared.covered
    result = answer_stream(prepared, history)
    assert "".join(result) == "Hello world"
    result.close()
    turn = finish_turn(prepared, result)
    assert turn.answer == "Hello world"
    assert turn.answer_record.output_tokens == 50
    assert turn.cost_usd > 0
    assert fake_chat.streams[0][0][-1].content.endswith("Question: What about kids?")
    log_turn(prepared, "answered")
    events = log_lines()
    turn_event = next(e for e in events if e["event"] == "turn")
    assert turn_event["outcome"] == "answered"
    assert turn_event["search_queries"] == ["swimming laps"]
    assert {e["kind"] for e in events if e["event"] == "llm_call"} == {"research", "answer"}
    assert all(e["session"] == "s" and e["turn"] == 2 for e in events)


def test_not_covered_turn(fake_chat):
    far = make_chunk(text="far")
    far = type(far)(**{**far.__dict__, "distance": 0.9})
    prepared = prepare_turn(
        "Capital of France?", [], None, FakeRetriever(chunks=[far]), session="s", turn_no=1
    )
    assert not prepared.covered
    turn = finish_turn(prepared, None)
    assert turn.answer == NOT_COVERED
    assert turn.answer_record is None
    assert turn.cost_usd == prepared.research.cost_usd


def test_stream_error_surfaces_as_llm_error(fake_chat):
    fake_chat.stream_error = RuntimeError("boom")
    prepared = prepare_turn("Hi there", [], None, FakeRetriever(), session="s", turn_no=1)
    result = answer_stream(prepared, [])
    with pytest.raises(LLMError):
        list(result)


def test_check_citations():
    chunks = [make_chunk(0), make_chunk(1, label="Dietary Guidelines", page_ref="p. 5")]
    ok, cites = check_citations(
        "Eat less (Dietary Guidelines, p. 5). Move more (Mini Guide, PDF p. 1).", chunks
    )
    assert ok and cites == [("Dietary Guidelines", "p. 5"), ("Mini Guide", "PDF p. 1")]
    ok, _ = check_citations("Wrong page (Dietary Guidelines, p. 9).", chunks)
    assert not ok
    ok, cites = check_citations("No citations here.", chunks)
    assert ok and cites == []
