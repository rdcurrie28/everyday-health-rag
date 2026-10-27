import hmac
import os
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL = "claude-haiku-4-5-20251001"
PRICE_INPUT_PER_MTOK = 1.00
PRICE_OUTPUT_PER_MTOK = 5.00

ANSWER_MAX_TOKENS = 700
RESEARCH_MAX_TOKENS = 200
RESEARCH_HISTORY_TURNS = 2
ANSWER_HISTORY_TURNS = 4
RESEARCH_ANSWER_CHARS = 300
MAX_SEARCHES = 3
MAX_QUERY_CHARS = 300
MAX_EXCERPTS = 6

K = 3
MAX_QUESTION_CHARS = 500
MAX_TURNS_PER_SESSION = 20

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_MAX_TOKENS = {EMBEDDING_MODEL: 512, MINILM_MODEL: 256}
RELEVANCE_MAX_DISTANCE = {EMBEDDING_MODEL: 0.45, MINILM_MODEL: 0.78}
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BASELINE_CHUNK_SIZE = 4000
MIN_CHUNKS_PER_SOURCE = 20

COLLECTION = "health_docs"
MANIFEST_NAME = "index_manifest.json"


@dataclass(frozen=True)
class Settings:
    app_mode: str
    data_dir: Path
    index_dir: Path
    model_cache_dir: Path
    corpus_file: Path
    git_sha: str
    anthropic_api_key: str | None = field(default=None, repr=False)
    app_password: str | None = field(default=None, repr=False)

    @property
    def deployed(self) -> bool:
        return self.app_mode == "deployed"

    def missing_secrets(self) -> list[str]:
        missing = []
        if not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if not self.app_password:
            missing.append("APP_PASSWORD")
        return missing


def _path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


@cache
def get_settings() -> Settings:
    app_mode = os.environ.get("APP_MODE", "local")
    if app_mode == "local":
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    return Settings(
        app_mode=app_mode,
        data_dir=_path("DATA_DIR", REPO_ROOT / "data"),
        index_dir=_path("INDEX_DIR", REPO_ROOT / ".index"),
        model_cache_dir=_path("MODEL_CACHE_DIR", REPO_ROOT / ".models"),
        corpus_file=_path("CORPUS_FILE", REPO_ROOT / "corpus.toml"),
        git_sha=os.environ.get("GIT_SHA", "unknown"),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        app_password=os.environ.get("APP_PASSWORD") or None,
    )


def check_password(entered: str, configured: str) -> bool:
    return hmac.compare_digest(entered.encode("utf-8"), configured.encode("utf-8"))
