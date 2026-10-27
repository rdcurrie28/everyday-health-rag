import dataclasses
import re

import pytest

from healthrag.build_index import (
    IndexBuildError,
    build,
    chunk_document,
    clear_target,
    page_ref,
    query_prefix_for,
)
from healthrag.config import BGE_QUERY_PREFIX, MANIFEST_NAME, MINILM_MODEL
from healthrag.corpus import load_corpus

from .conftest import FIXTURES, fixture_settings


@pytest.fixture
def mini_doc():
    return load_corpus(FIXTURES / "corpus.toml", FIXTURES)[0]


def test_page_ref_uses_printed_label_only_when_it_differs():
    assert page_ref(0, "1") == "PDF p. 1"
    assert page_ref(1, "ii") == "p. ii"
    assert page_ref(4, None) == "PDF p. 5"


def test_query_prefix_only_for_bge():
    assert query_prefix_for("BAAI/bge-small-en-v1.5") == BGE_QUERY_PREFIX
    assert query_prefix_for(MINILM_MODEL) == ""


def test_chunks_cover_both_page_ref_branches(mini_doc):
    chunks = chunk_document(mini_doc)
    page1 = [c for c in chunks if c.metadata["pdf_page"] == 1]
    page2 = [c for c in chunks if c.metadata["pdf_page"] == 2]
    assert page1 and page2
    assert all(c.page_content.startswith("Mini Guide · PDF p. 1\n") for c in page1)
    assert all(c.page_content.startswith("Mini Guide · p. ii\n") for c in page2)
    assert all(re.fullmatch(r"mini-p1-\d+", c.id) for c in page1)
    assert all(c.metadata["source"] == "mini" for c in chunks)
    assert len(chunks) >= 4


def test_skip_pages_drops_pages(mini_doc):
    doc = dataclasses.replace(mini_doc, skip_pages=frozenset({2}))
    assert {c.metadata["pdf_page"] for c in chunk_document(doc)} == {1}


def test_baseline_has_no_prefix_and_no_skipping(mini_doc):
    doc = dataclasses.replace(mini_doc, skip_pages=frozenset({2}))
    chunks = chunk_document(doc, baseline=True)
    assert {c.metadata["pdf_page"] for c in chunks} == {1, 2}
    assert not any(c.page_content.startswith("Mini Guide ·") for c in chunks)


def test_manifest_contents(mini_index):
    _, manifest = mini_index
    assert manifest["sources"]["mini"]["label"] == "Mini Guide"
    assert manifest["sources"]["mini"]["chunks"] >= 4
    assert manifest["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert manifest["embedding_dim"] == 384
    assert manifest["query_prefix"] == BGE_QUERY_PREFIX
    assert manifest["approx_tokens_max"] > 0
    assert {"chunk_size", "chunk_overlap", "libraries", "built_at"} <= manifest.keys()


def test_min_chunks_failure(tmp_path, mini_doc):
    settings = fixture_settings(tmp_path / "idx")
    with pytest.raises(IndexBuildError, match="at least 999"):
        build(settings, [mini_doc], min_chunks=999)


def test_rebuild_does_not_duplicate(tmp_path, mini_doc):
    settings = fixture_settings(tmp_path / "idx")
    first = build(settings, [mini_doc], min_chunks=1)
    second = build(settings, [mini_doc], min_chunks=1)
    assert first["sources"]["mini"]["chunks"] == second["sources"]["mini"]["chunks"]
    assert (settings.index_dir / MANIFEST_NAME).is_file()


def test_clear_target_refuses_unknown_folder(tmp_path):
    folder = tmp_path / "precious"
    folder.mkdir()
    (folder / "notes.txt").write_text("keep me")
    with pytest.raises(IndexBuildError, match="delete it by hand"):
        clear_target(folder)
    assert (folder / "notes.txt").exists()


def test_clear_target_removes_empty_or_index_folder(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    clear_target(empty)
    assert not empty.exists()
    index = tmp_path / "index"
    index.mkdir()
    (index / MANIFEST_NAME).write_text("{}")
    clear_target(index)
    assert not index.exists()
