import dataclasses
import time
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage

from healthrag.config import MODEL, PRICE_INPUT_PER_MTOK, PRICE_OUTPUT_PER_MTOK
from healthrag.logs import log_event

BUSY_TYPES = {"rate_limit_error", "overloaded_error", "api_error", "timeout_error"}
BUSY_CLASSES = tuple(
    cls
    for cls in (
        getattr(anthropic, "RateLimitError", None),
        getattr(anthropic, "OverloadedError", None),
        getattr(anthropic, "InternalServerError", None),
        getattr(anthropic, "ServiceUnavailableError", None),
        getattr(anthropic, "DeadlineExceededError", None),
        getattr(anthropic, "APIConnectionError", None),
        getattr(anthropic, "APITimeoutError", None),
    )
    if cls is not None
)


@dataclass
class CallRecord:
    kind: str
    model: str = MODEL
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int = 0
    input_tokens_estimated: bool = False
    aborted: bool = False
    error: str | None = None
    id: str | None = None


class LLMError(Exception):
    def __init__(self, kind: str, detail: str, record: CallRecord | None = None):
        super().__init__(detail)
        self.kind = kind
        self.detail = detail
        self.record = record


def error_type(e: BaseException) -> str | None:
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("type"), str):
            return error["type"]
    return None


def classify(e: BaseException) -> LLMError:
    etype = error_type(e)
    if etype is not None:
        kind = "busy" if etype in BUSY_TYPES else "unavailable"
    else:
        kind = "busy" if isinstance(e, BUSY_CLASSES) else "unavailable"
    return LLMError(kind, etype or type(e).__name__)


def cost(input_tokens: int | None, output_tokens: int | None) -> float | None:
    if input_tokens is None and output_tokens is None:
        return None
    return round(
        (input_tokens or 0) * PRICE_INPUT_PER_MTOK / 1_000_000
        + (output_tokens or 0) * PRICE_OUTPUT_PER_MTOK / 1_000_000,
        6,
    )


def estimate_tokens(messages: list[BaseMessage]) -> int:
    return sum(len(str(message.content)) for message in messages) // 4


def message_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


@cache
def _client() -> ChatAnthropic:
    return ChatAnthropic(model=MODEL, max_retries=2, stream_usage=True)


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _log(record: CallRecord, session: str, turn: int) -> None:
    log_event("llm_call", **dataclasses.asdict(record), session=session, turn=turn)


def invoke(
    kind: str,
    messages: list[BaseMessage],
    *,
    max_tokens: int,
    session: str,
    turn: int,
    temperature: float = 0.0,
    tools: list[dict] | None = None,
) -> tuple[AIMessage, CallRecord]:
    record = CallRecord(kind=kind)
    start = time.perf_counter()
    model = _client().bind_tools(tools) if tools else _client()
    try:
        response = model.invoke(messages, max_tokens=max_tokens, temperature=temperature)
    except Exception as e:
        error = classify(e)
        record.error = error.detail
        record.id = getattr(e, "request_id", None)
        record.latency_ms = _elapsed_ms(start)
        _log(record, session, turn)
        error.record = record
        raise error from e
    usage = response.usage_metadata or {}
    record.input_tokens = usage.get("input_tokens")
    record.output_tokens = usage.get("output_tokens")
    record.cost_usd = cost(record.input_tokens, record.output_tokens)
    record.id = response.id or response.response_metadata.get("id")
    record.latency_ms = _elapsed_ms(start)
    _log(record, session, turn)
    return response, record


class StreamResult:
    def __init__(
        self,
        kind: str,
        messages: list[BaseMessage],
        *,
        max_tokens: int,
        session: str,
        turn: int,
        temperature: float = 0.0,
    ):
        self.kind = kind
        self.session = session
        self.turn = turn
        self.text = ""
        self.ttft_ms: int | None = None
        self.record = CallRecord(kind=kind)
        self._messages = messages
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._completed = False
        self._gen: Iterator[str] | None = None

    def __iter__(self) -> Iterator[str]:
        if self._gen is None:
            self._gen = self._run()
        return self._gen

    def _run(self) -> Iterator[str]:
        start = time.perf_counter()
        parts: list[str] = []
        aggregate = None
        inner = None
        try:
            inner = iter(
                _client().stream(
                    self._messages, max_tokens=self._max_tokens, temperature=self._temperature
                )
            )
            for chunk in inner:
                aggregate = chunk if aggregate is None else aggregate + chunk
                piece = message_text(chunk.content)
                if piece:
                    if self.ttft_ms is None:
                        self.ttft_ms = _elapsed_ms(start)
                    parts.append(piece)
                    yield piece
            self._completed = True
        except Exception as e:
            error = classify(e)
            self.record.error = error.detail
            self.record.id = getattr(e, "request_id", None)
            error.record = self.record
            raise error from e
        finally:
            if not self._completed and inner is not None and hasattr(inner, "close"):
                inner.close()
            self.text = "".join(parts)
            self._finish(aggregate, start)

    def _finish(self, aggregate, start: float) -> None:
        usage = getattr(aggregate, "usage_metadata", None) if aggregate is not None else None
        if self._completed and usage:
            self.record.input_tokens = usage.get("input_tokens")
            self.record.output_tokens = usage.get("output_tokens")
        else:
            self.record.aborted = self.record.error is None
            self.record.input_tokens = estimate_tokens(self._messages)
            self.record.input_tokens_estimated = True
        self.record.cost_usd = cost(self.record.input_tokens, self.record.output_tokens)
        self.record.latency_ms = _elapsed_ms(start)
        _log(self.record, self.session, self.turn)

    def close(self) -> None:
        if self._gen is not None:
            self._gen.close()


def stream(
    kind: str,
    messages: list[BaseMessage],
    *,
    max_tokens: int,
    session: str,
    turn: int,
    temperature: float = 0.0,
) -> StreamResult:
    return StreamResult(
        kind, messages, max_tokens=max_tokens, session=session, turn=turn, temperature=temperature
    )
