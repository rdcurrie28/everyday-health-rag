import sys

import pytest

from healthrag import serve
from healthrag.retrieval import IndexNotReady


@pytest.fixture
def launch(monkeypatch):
    calls = []
    monkeypatch.setattr(serve, "get_settings", lambda: "settings")
    monkeypatch.setattr(serve.cli, "main", lambda: calls.append(list(sys.argv)) or 0)
    monkeypatch.setattr(sys, "argv", ["serve", "--server.port=9000"])

    def run(load):
        monkeypatch.setattr(serve, "shared_retriever", load)
        with pytest.raises(SystemExit) as exit_info:
            serve.main()
        assert exit_info.value.code == 0
        return calls

    return run


def test_loads_index_before_starting_streamlit(launch):
    loaded = []
    calls = launch(lambda settings: loaded.append(settings))
    assert loaded == ["settings"]
    assert calls == [["streamlit", "run", "app.py", "--server.port=9000"]]


def test_missing_index_still_starts_streamlit(launch):
    def missing(settings):
        raise IndexNotReady("Search index not found")

    assert launch(missing) == [["streamlit", "run", "app.py", "--server.port=9000"]]
