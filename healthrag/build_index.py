import argparse
import dataclasses
import json
import shutil
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from statistics import quantiles

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from healthrag.config import (
    BASELINE_CHUNK_SIZE,
    BGE_QUERY_PREFIX,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION,
    EMBEDDING_MODEL,
    MANIFEST_NAME,
    MIN_CHUNKS_PER_SOURCE,
    MODEL_MAX_TOKENS,
    Settings,
    get_settings,
)
from healthrag.corpus import CorpusDoc, CorpusError, load_corpus
from healthrag.logs import log_event

LIBRARIES = ("langchain-core", "langchain-community", "langchain-chroma", "chromadb", "fastembed")


class IndexBuildError(RuntimeError):
    pass


def page_ref(page_index: int, page_label: str | None) -> str:
    pdf_page = page_index + 1
    if page_label and page_label != str(pdf_page):
        return f"p. {page_label}"
    return f"PDF p. {pdf_page}"


def query_prefix_for(embedding_model: str) -> str:
    return BGE_QUERY_PREFIX if embedding_model.startswith("BAAI/bge") else ""


def clear_target(index_dir: Path) -> None:
    if not index_dir.exists():
        return
    if index_dir.is_dir() and (
        not any(index_dir.iterdir()) or (index_dir / MANIFEST_NAME).exists()
    ):
        shutil.rmtree(index_dir)
        return
    raise IndexBuildError(
        f"Refusing to delete {index_dir}: it is not empty and has no {MANIFEST_NAME}. "
        "If it is left over from a failed build, delete it by hand."
    )


def chunk_document(doc: CorpusDoc, *, baseline: bool = False) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=BASELINE_CHUNK_SIZE if baseline else CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks: list[Document] = []
    for page in PyPDFLoader(str(doc.path)).load():
        page_index = page.metadata["page"]
        pdf_page = page_index + 1
        if not baseline and pdf_page in doc.skip_pages:
            continue
        text = page.page_content.strip()
        if not text:
            continue
        ref = page_ref(page_index, page.metadata.get("page_label"))
        for n, piece in enumerate(splitter.split_text(text)):
            chunks.append(
                Document(
                    id=f"{doc.id}-p{pdf_page}-{n}",
                    page_content=piece if baseline else f"{doc.label} · {ref}\n{piece}",
                    metadata={
                        "source": doc.id,
                        "label": doc.label,
                        "pdf_page": pdf_page,
                        "page_ref": ref,
                    },
                )
            )
    return chunks


def build(
    settings: Settings,
    docs: list[CorpusDoc],
    *,
    embedding_model: str = EMBEDDING_MODEL,
    min_chunks: int = MIN_CHUNKS_PER_SOURCE,
    baseline: bool = False,
) -> dict:
    index_dir = settings.index_dir
    clear_target(index_dir)

    chunks_by_source = {doc.id: chunk_document(doc, baseline=baseline) for doc in docs}
    for doc in docs:
        count = len(chunks_by_source[doc.id])
        if count < min_chunks:
            raise IndexBuildError(
                f"Source '{doc.id}' produced {count} chunks; at least {min_chunks} are required."
            )
    all_chunks = [chunk for chunks in chunks_by_source.values() for chunk in chunks]

    embeddings = FastEmbedEmbeddings(
        model_name=embedding_model, cache_dir=str(settings.model_cache_dir)
    )
    index_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(index_dir), settings=ChromaSettings(anonymized_telemetry=False)
    )
    prefix = query_prefix_for(embedding_model)
    try:
        store = Chroma(
            client=client,
            collection_name=COLLECTION,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
        store.add_documents(all_chunks, ids=[chunk.id for chunk in all_chunks])
        store.similarity_search(prefix + "physical activity for adults", k=1)
    finally:
        client.close()
    embedding_dim = len(embeddings.embed_query(prefix + "warm-up"))

    approx_tokens = [len(chunk.page_content) // 4 for chunk in all_chunks]
    approx_max = max(approx_tokens)
    approx_p95 = quantiles(approx_tokens, n=20)[-1] if len(approx_tokens) > 1 else approx_max
    limit = MODEL_MAX_TOKENS.get(embedding_model)
    if limit and approx_max > limit:
        log_event(
            "chunk_too_long",
            embedding_model=embedding_model,
            approx_tokens_max=approx_max,
            model_limit=limit,
        )

    manifest = {
        "sources": {
            doc.id: {
                "label": doc.label,
                "edition": doc.edition,
                "url": doc.url,
                "chunks": len(chunks_by_source[doc.id]),
            }
            for doc in docs
        },
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "query_prefix": prefix,
        "chunk_size": BASELINE_CHUNK_SIZE if baseline else CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "baseline": baseline,
        "approx_tokens_max": approx_max,
        "approx_tokens_p95": round(approx_p95, 1),
        "libraries": {name: version(name) for name in LIBRARIES},
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (index_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the search index from corpus.toml.")
    parser.add_argument("--index-dir", type=Path, help="Write the index here instead of INDEX_DIR.")
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="4,000-character chunks, no skipped pages, no label prefix (for the eval baseline).",
    )
    parser.add_argument(
        "--min-chunks",
        type=int,
        help=f"Minimum chunks per source (default {MIN_CHUNKS_PER_SOURCE}; 1 with --baseline).",
    )
    args = parser.parse_args(argv)
    min_chunks = args.min_chunks or (1 if args.baseline else MIN_CHUNKS_PER_SOURCE)

    settings = get_settings()
    if args.index_dir:
        settings = dataclasses.replace(settings, index_dir=args.index_dir)
    try:
        docs = load_corpus(settings.corpus_file, settings.data_dir)
        manifest = build(
            settings,
            docs,
            embedding_model=args.embedding_model,
            min_chunks=min_chunks,
            baseline=args.baseline,
        )
    except (CorpusError, IndexBuildError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    log_event(
        "index_built",
        index_dir=settings.index_dir,
        embedding_model=manifest["embedding_model"],
        baseline=manifest["baseline"],
        chunks={sid: info["chunks"] for sid, info in manifest["sources"].items()},
        approx_tokens_max=manifest["approx_tokens_max"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
