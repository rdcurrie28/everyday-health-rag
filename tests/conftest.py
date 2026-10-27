import dataclasses
import json
import warnings
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from healthrag import llm
from healthrag.build_index import build
from healthrag.config import REPO_ROOT, get_settings
from healthrag.corpus import load_corpus
from healthrag.retrieval import Chunk, open_retriever

warnings.filterwarnings("ignore", category=DeprecationWarning)

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_settings(index_dir: Path):
    return dataclasses.replace(
        get_settings(),
        data_dir=FIXTURES,
        corpus_file=FIXTURES / "corpus.toml",
        index_dir=index_dir,
        model_cache_dir=REPO_ROOT / ".models",
    )


@pytest.fixture(scope="session")
def mini_index(tmp_path_factory):
    settings = fixture_settings(tmp_path_factory.mktemp("index") / "idx")
    docs = load_corpus(settings.corpus_file, settings.data_dir)
    manifest = build(settings, docs, min_chunks=1)
    return settings, manifest


@pytest.fixture(scope="session")
def retriever(mini_index):
    settings, _ = mini_index
    return open_retriever(settings)


def make_chunk(
    n: int = 0, label: str = "Mini Guide", page_ref: str = "PDF p. 1", text: str = "text"
) -> Chunk:
    return Chunk(
        id=f"mini-p1-{n}", source="mini", label=label, page_ref=page_ref, text=text, distance=0.1
    )


class FakeRetriever:
    max_distance = 0.45

    def __init__(self, chunks=None):
        self.chunks = chunks if chunks is not None else [make_chunk()]
        self.calls: list[tuple[str, str | None]] = []

    def search(self, query, source=None):
        self.calls.append((query, source))
        return self.chunks

    def relevant(self, chunks):
        return [chunk for chunk in chunks if chunk.distance <= self.max_distance]


USAGE = {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}


def tool_reply(*queries: str, guide: str | None = None) -> AIMessage:
    calls = [
        {
            "name": "search_guides",
            "args": {"query": query, **({"guide": guide} if guide else {})},
            "id": f"toolu_{i}",
            "type": "tool_call",
        }
        for i, query in enumerate(queries)
    ]
    return AIMessage(content="", tool_calls=calls, id="msg_tool", usage_metadata=USAGE)


def done_reply(text: str = "DONE") -> AIMessage:
    return AIMessage(content=text, id="msg_done", usage_metadata=USAGE)


class FakeChat:
    def __init__(self, stream_pieces=("Hello", " world")):
        self.responses: list[AIMessage] = [tool_reply("swimming laps"), done_reply()]
        self.invoke_error: BaseException | None = None
        self.stream_pieces = list(stream_pieces)
        self.stream_error: BaseException | None = None
        self.stream_error_after = 0
        self.bound_tools: list = []
        self.invocations: list = []
        self.streams: list = []

    def bind_tools(self, tools):
        self.bound_tools.append(tools)
        return self

    def invoke(self, messages, **kwargs):
        self.invocations.append((list(messages), kwargs))
        if self.invoke_error is not None:
            raise self.invoke_error
        if self.responses:
            return self.responses.pop(0)
        return done_reply()

    def stream(self, messages, **kwargs):
        self.streams.append((messages, kwargs))
        for i, piece in enumerate(self.stream_pieces):
            if self.stream_error is not None and i == self.stream_error_after:
                raise self.stream_error
            yield AIMessageChunk(content=piece)
        if self.stream_error is not None and self.stream_error_after >= len(self.stream_pieces):
            raise self.stream_error
        yield AIMessageChunk(
            content="",
            usage_metadata={"input_tokens": 1000, "output_tokens": 50, "total_tokens": 1050},
        )


@pytest.fixture
def fake_chat(monkeypatch):
    chat = FakeChat()
    monkeypatch.setattr(llm, "_client", lambda: chat)
    return chat


@pytest.fixture
def log_lines(capsys):
    def read():
        lines = []
        for line in capsys.readouterr().out.splitlines():
            if line.startswith("{"):
                lines.append(json.loads(line))
        return lines

    return read
