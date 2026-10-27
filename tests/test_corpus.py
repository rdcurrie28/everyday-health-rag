import pytest

from healthrag.corpus import CorpusError, load_corpus, parse_pages

from .conftest import FIXTURES


def write_corpus(tmp_path, body: str):
    path = tmp_path / "corpus.toml"
    path.write_text(body, encoding="utf-8")
    return path


ENTRY = """
[[doc]]
id = "{id}"
label = "Mini Guide"
file = "{file}"
url = "https://example.invalid"
edition = "1"
skip_pages = {skip}
"""


def test_valid_corpus_loads():
    docs = load_corpus(FIXTURES / "corpus.toml", FIXTURES)
    assert [doc.id for doc in docs] == ["mini"]
    assert docs[0].path == FIXTURES / "mini.pdf"
    assert docs[0].skip_pages == frozenset()


def test_page_specs_expand():
    assert parse_pages(["9", "1-3"], "x", "c.toml") == frozenset({1, 2, 3, 9})


@pytest.mark.parametrize("spec", [["a"], ["1-b"], ["3-1"], ["0"]])
def test_bad_page_specs_raise(spec):
    with pytest.raises(CorpusError):
        parse_pages(spec, "x", "c.toml")


def test_duplicate_ids_raise(tmp_path):
    body = ENTRY.format(id="a", file="mini.pdf", skip="[]") * 2
    with pytest.raises(CorpusError, match="duplicate"):
        load_corpus(write_corpus(tmp_path, body), FIXTURES)


def test_missing_file_raises(tmp_path):
    body = ENTRY.format(id="a", file="nope.pdf", skip="[]")
    with pytest.raises(CorpusError, match="not found"):
        load_corpus(write_corpus(tmp_path, body), FIXTURES)


def test_missing_keys_raise(tmp_path):
    with pytest.raises(CorpusError, match="missing"):
        load_corpus(write_corpus(tmp_path, '[[doc]]\nid = "a"\n'), FIXTURES)


def test_empty_corpus_raises(tmp_path):
    with pytest.raises(CorpusError, match="no"):
        load_corpus(write_corpus(tmp_path, ""), FIXTURES)
