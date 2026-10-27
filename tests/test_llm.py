import importlib
import json
import sys

import anthropic
import httpx
import pytest
from langchain_core.messages import HumanMessage

from healthrag import llm
from healthrag.llm import CallRecord, LLMError, StreamResult, classify, cost, estimate_tokens

from .conftest import done_reply

REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def status_error(cls, status: int, body):
    return cls("err", response=httpx.Response(status, request=REQUEST), body=body)


def api_body(error_type: str) -> dict:
    return {"type": "error", "error": {"type": error_type, "message": "x"}}


@pytest.mark.parametrize(
    ("error_type", "kind"),
    [
        ("rate_limit_error", "busy"),
        ("overloaded_error", "busy"),
        ("api_error", "busy"),
        ("timeout_error", "busy"),
        ("authentication_error", "unavailable"),
        ("permission_error", "unavailable"),
        ("invalid_request_error", "unavailable"),
        ("billing_error", "unavailable"),
        ("something_new", "unavailable"),
    ],
)
def test_classify_by_error_type(error_type, kind):
    error = classify(status_error(anthropic.APIStatusError, 400, api_body(error_type)))
    assert (error.kind, error.detail) == (kind, error_type)


def test_mid_stream_overloaded_with_status_200_is_busy():
    error = classify(status_error(anthropic.APIStatusError, 200, api_body("overloaded_error")))
    assert error.kind == "busy"


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (status_error(anthropic.RateLimitError, 429, None), "busy"),
        (
            status_error(anthropic.InternalServerError, 503, "<html>Service Unavailable</html>"),
            "busy",
        ),
        (anthropic.APIConnectionError(request=REQUEST), "busy"),
        (anthropic.APITimeoutError(request=REQUEST), "busy"),
        (status_error(anthropic.AuthenticationError, 401, "not json"), "unavailable"),
        (ValueError("bug"), "unavailable"),
    ],
)
def test_classify_without_error_type(exc, kind):
    error = classify(exc)
    assert error.kind == kind
    assert error.detail == type(exc).__name__


def test_error_type_ignores_malformed_bodies():
    assert llm.error_type(status_error(anthropic.APIStatusError, 400, {"error": "flat"})) is None
    assert (
        llm.error_type(status_error(anthropic.APIStatusError, 400, {"error": {"type": 3}})) is None
    )


def test_message_text_joins_text_blocks():
    content = ["a", {"type": "text", "text": "b"}, {"type": "tool_use", "id": "x"}]
    assert llm.message_text(content) == "ab"
    assert llm.message_text(None) == ""


def test_cost():
    assert cost(None, None) is None
    assert cost(1_000_000, 0) == pytest.approx(1.0)
    assert cost(1000, 100) == pytest.approx(0.0015)
    assert cost(4000, None) == pytest.approx(0.004)


def test_estimate_tokens():
    assert estimate_tokens([HumanMessage("x" * 400)]) == 100


def test_importing_llm_creates_no_client(monkeypatch):
    monkeypatch.delitem(sys.modules, "healthrag.llm", raising=False)
    module = importlib.import_module("healthrag.llm")
    assert module._client.cache_info().currsize == 0


def test_real_client_settings(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-dummy")
    llm._client.cache_clear()
    try:
        client = llm._client()
        assert client.max_retries == 2
        assert client.stream_usage is True
        assert client.model == llm.MODEL
    finally:
        llm._client.cache_clear()


def test_invoke_records_usage_and_logs(fake_chat, log_lines):
    fake_chat.responses = [done_reply("plain text")]
    message, record = llm.invoke(
        "research", [HumanMessage("hi")], max_tokens=50, session="s", turn=3
    )
    assert llm.message_text(message.content) == "plain text"
    assert (record.input_tokens, record.output_tokens, record.id) == (100, 10, "msg_done")
    assert fake_chat.invocations[0][1] == {"max_tokens": 50, "temperature": 0.0}
    assert fake_chat.bound_tools == []
    (event,) = log_lines()
    assert event["event"] == "llm_call" and event["session"] == "s" and event["turn"] == 3
    assert event["cost_usd"] == pytest.approx(0.00015)


def test_invoke_with_tools_binds_them(fake_chat):
    tool = {"name": "t", "description": "d", "input_schema": {"type": "object"}}
    message, _ = llm.invoke(
        "research", [HumanMessage("hi")], max_tokens=50, session="s", turn=1, tools=[tool]
    )
    assert fake_chat.bound_tools == [[tool]]
    assert message.tool_calls[0]["name"] == "search_guides"


def test_invoke_error_raises_with_record(fake_chat, log_lines):
    fake_chat.invoke_error = status_error(
        anthropic.RateLimitError, 429, api_body("rate_limit_error")
    )
    with pytest.raises(LLMError) as info:
        llm.invoke("rewrite", [HumanMessage("hi")], max_tokens=50, session="s", turn=1)
    assert info.value.kind == "busy"
    assert info.value.record.error == "rate_limit_error"
    assert log_lines()[0]["error"] == "rate_limit_error"


def stream(**kwargs) -> StreamResult:
    return llm.stream("answer", [HumanMessage("x" * 400)], max_tokens=100, session="s", turn=1)


def test_completed_stream_reports_real_usage(fake_chat, log_lines):
    result = stream()
    assert list(result) == ["Hello", " world"]
    result.close()
    assert result.text == "Hello world"
    assert result.record.input_tokens == 1000 and result.record.output_tokens == 50
    assert not result.record.aborted and not result.record.input_tokens_estimated
    assert result.ttft_ms is not None
    events = [e for e in log_lines() if e["event"] == "llm_call"]
    assert len(events) == 1 and events[0]["input_tokens"] > 0


def test_closed_stream_logs_one_aborted_record(fake_chat, log_lines):
    result = stream()
    iterator = iter(result)
    assert next(iterator) == "Hello"
    result.close()
    result.close()
    events = [e for e in log_lines() if e["event"] == "llm_call"]
    assert len(events) == 1
    assert events[0]["aborted"] is True
    assert events[0]["input_tokens_estimated"] is True
    assert events[0]["input_tokens"] == 100
    assert events[0]["output_tokens"] is None
    assert result.text == "Hello"


def test_mid_stream_error_logs_once_and_raises(fake_chat, log_lines):
    fake_chat.stream_error = status_error(
        anthropic.APIStatusError, 200, api_body("overloaded_error")
    )
    fake_chat.stream_error_after = 1
    result = stream()
    with pytest.raises(LLMError) as info:
        list(result)
    result.close()
    assert info.value.kind == "busy"
    events = [e for e in log_lines() if e["event"] == "llm_call"]
    assert len(events) == 1
    assert events[0]["error"] == "overloaded_error" and events[0]["aborted"] is False


def test_iterating_again_never_starts_a_second_stream(fake_chat):
    result = stream()
    assert iter(result) is iter(result)
    assert list(result) == ["Hello", " world"]
    assert list(result) == []
    assert len(fake_chat.streams) == 1


def test_close_before_iteration_is_safe(fake_chat, log_lines):
    stream().close()
    assert log_lines() == []


def test_log_lines_have_no_secrets(fake_chat, capsys):
    llm.invoke("rewrite", [HumanMessage("hi")], max_tokens=5, session="s", turn=1)
    line = capsys.readouterr().out.strip()
    record = json.loads(line)
    assert "api_key" not in record and "password" not in record
    assert set(CallRecord.__dataclass_fields__) <= record.keys()
