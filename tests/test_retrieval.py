import json

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from healthrag.config import COLLECTION, MANIFEST_NAME, K
from healthrag.retrieval import IndexNotReady, Retriever, open_retriever

from .conftest import fixture_settings


def test_missing_manifest(tmp_path):
    with pytest.raises(IndexNotReady, match="not found"):
        open_retriever(fixture_settings(tmp_path))


def test_missing_collection_is_not_created(tmp_path):
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"embedding_model": "x", "sources": {}}))
    with pytest.raises(IndexNotReady):
        open_retriever(fixture_settings(tmp_path))
    client = chromadb.PersistentClient(
        path=str(tmp_path), settings=ChromaSettings(anonymized_telemetry=False)
    )
    assert COLLECTION not in [c.name for c in client.list_collections()]


def test_empty_collection(tmp_path):
    (tmp_path / MANIFEST_NAME).write_text(json.dumps({"embedding_model": "x", "sources": {}}))
    client = chromadb.PersistentClient(
        path=str(tmp_path), settings=ChromaSettings(anonymized_telemetry=False)
    )
    client.create_collection(COLLECTION)
    with pytest.raises(IndexNotReady, match="empty"):
        open_retriever(fixture_settings(tmp_path))


def test_search_returns_k_chunks_with_page_refs(retriever):
    chunks = retriever.search("swimming laps")
    assert len(chunks) == K
    assert all(chunk.page_ref in {"PDF p. 1", "p. ii"} for chunk in chunks)
    assert chunks[0].page_ref == "p. ii"
    assert chunks[0].id.startswith("mini-p2-")


def test_source_filter(retriever):
    assert retriever.search("swimming", source="other") == []
    assert all(c.source == "mini" for c in retriever.search("swimming", source="mini"))


def test_labels_and_manifest(retriever):
    assert retriever.labels == {"mini": "Mini Guide"}
    assert retriever.manifest["sources"]["mini"]["edition"] == "test edition"


class RecordingStore:
    def __init__(self):
        self.queries = []

    def similarity_search_with_score(self, query, k, filter=None):
        self.queries.append(query)
        return []


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [("Represent this: ", "Represent this: walking"), ("", "walking")],
)
def test_query_prefix_applied_from_manifest(prefix, expected):
    store = RecordingStore()
    Retriever(store, {"query_prefix": prefix, "sources": {}}).search("walking")
    assert store.queries == [expected]
