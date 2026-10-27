import json
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

from healthrag.config import COLLECTION, MANIFEST_NAME, RELEVANCE_MAX_DISTANCE, K, Settings


class IndexNotReady(RuntimeError):
    pass


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    label: str
    page_ref: str
    text: str
    distance: float


class Retriever:
    def __init__(self, store: Chroma, manifest: dict):
        self._store = store
        self._manifest = manifest

    @property
    def manifest(self) -> dict:
        return self._manifest

    @property
    def labels(self) -> dict[str, str]:
        return {sid: info["label"] for sid, info in self._manifest["sources"].items()}

    @property
    def max_distance(self) -> float:
        return RELEVANCE_MAX_DISTANCE.get(self._manifest.get("embedding_model", ""), 1.0)

    def relevant(self, chunks: list[Chunk]) -> list[Chunk]:
        return [chunk for chunk in chunks if chunk.distance <= self.max_distance]

    def search(self, query: str, source: str | None = None) -> list[Chunk]:
        results = self._store.similarity_search_with_score(
            self._manifest.get("query_prefix", "") + query,
            k=K,
            filter={"source": source} if source else None,
        )
        return [
            Chunk(
                id=doc.id or "",
                source=doc.metadata["source"],
                label=doc.metadata["label"],
                page_ref=doc.metadata["page_ref"],
                text=doc.page_content,
                distance=float(score),
            )
            for doc, score in results
        ]


def open_retriever(settings: Settings) -> Retriever:
    manifest_path = settings.index_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise IndexNotReady(f"Search index not found at {settings.index_dir}.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    client = chromadb.PersistentClient(
        path=str(settings.index_dir), settings=ChromaSettings(anonymized_telemetry=False)
    )
    try:
        collection = client.get_collection(COLLECTION)
    except (NotFoundError, ValueError) as e:
        raise IndexNotReady(f"Search index not found at {settings.index_dir}.") from e
    if collection.count() == 0:
        raise IndexNotReady(f"Search index at {settings.index_dir} is empty.")

    embeddings = FastEmbedEmbeddings(
        model_name=manifest["embedding_model"], cache_dir=str(settings.model_cache_dir)
    )
    store = Chroma(
        client=client,
        collection_name=COLLECTION,
        embedding_function=embeddings,
        create_collection_if_not_exists=False,
    )
    return Retriever(store, manifest)
