import tomllib
from dataclasses import dataclass
from pathlib import Path

REQUIRED_KEYS = ("id", "label", "file", "url", "edition")


class CorpusError(ValueError):
    pass


@dataclass(frozen=True)
class CorpusDoc:
    id: str
    label: str
    path: Path
    url: str
    edition: str
    skip_pages: frozenset[int]


def parse_pages(spec: list, doc_id: str, corpus_path: Path) -> frozenset[int]:
    pages: set[int] = set()
    for item in spec:
        text = str(item).strip()
        try:
            if "-" in text:
                first, last = (int(part) for part in text.split("-", 1))
            else:
                first = last = int(text)
        except ValueError as e:
            raise CorpusError(
                f"{corpus_path}: doc '{doc_id}' has a malformed page range '{item}'"
            ) from e
        if first < 1 or last < first:
            raise CorpusError(
                f"{corpus_path}: doc '{doc_id}' has an invalid or reversed page range '{item}'"
            )
        pages.update(range(first, last + 1))
    return frozenset(pages)


def load_corpus(path: Path, data_dir: Path) -> list[CorpusDoc]:
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise CorpusError(f"{path}: {e}") from e

    docs: list[CorpusDoc] = []
    seen: set[str] = set()
    for entry in raw.get("doc", []):
        missing = [key for key in REQUIRED_KEYS if key not in entry]
        if missing:
            raise CorpusError(f"{path}: doc entry is missing {', '.join(missing)}")
        doc_id = entry["id"]
        if doc_id in seen:
            raise CorpusError(f"{path}: duplicate doc id '{doc_id}'")
        seen.add(doc_id)
        pdf = Path(data_dir) / entry["file"]
        if not pdf.is_file():
            raise CorpusError(f"{path}: doc '{doc_id}' file not found: {pdf}")
        docs.append(
            CorpusDoc(
                id=doc_id,
                label=entry["label"],
                path=pdf,
                url=entry["url"],
                edition=entry["edition"],
                skip_pages=parse_pages(entry.get("skip_pages", []), doc_id, path),
            )
        )
    if not docs:
        raise CorpusError(f"{path}: no [[doc]] entries")
    return docs
