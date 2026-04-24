"""
Configuration for the Banking Knowledge AI System.
Switch between dev (Groq + local Ollama) and prod (fully on-prem vLLM).
"""

import os
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str  # "groq", "ollama", "openai_compatible"
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2048


@dataclass
class EmbedConfig:
    """Embedding model configuration."""
    provider: str  # "ollama", "local"
    model: str
    tokenizer_model: Optional[str] = None
    base_url: str = "http://localhost:11434"
    dimensions: int = 768


@dataclass
class VectorDBConfig:
    """Qdrant vector database configuration."""
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "banking_chunks"
    use_in_memory: bool = True  # True for dev, False for Docker deployment


@dataclass
class GraphConfig:
    """Knowledge graph configuration."""
    backend: str = "networkx"  # "networkx" for dev, "neo4j" for prod
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"
    max_hops: int = 2
    similarity_threshold: float = 0.85


@dataclass
class RetrievalConfig:
    """Hybrid retrieval scoring weights."""
    top_k: int = 10
    rerank_top_k: int = 5
    alpha: float = 0.4   # embedding similarity weight
    beta: float = 0.4    # graph connectivity weight
    gamma: float = 0.2   # policy compliance weight
    evidence_threshold: float = 0.3  # below this, abstain
    min_evidence_nodes: int = 2


@dataclass
class ChunkingConfig:
    """Document chunking configuration."""
    max_chunk_tokens: int = 512
    overlap_tokens: int = 50
    preserve_tables: bool = True
    preserve_sections: bool = True


@dataclass
class Config:
    """Master configuration."""
    env: str = "dev"
    main_llm: LLMConfig = None
    fast_llm: LLMConfig = None
    embedding: EmbedConfig = None
    vector_db: VectorDBConfig = None
    graph: GraphConfig = None
    retrieval: RetrievalConfig = None
    chunking: ChunkingConfig = None
    data_dir: str = "data/raw_docs"
    parsed_dir: str = "data/parsed"
    graph_dir: str = "data/graph"

    def __post_init__(self):
        if self.env == "dev":
            self._init_dev()
        else:
            self._init_prod()

        self._apply_offline_overrides_if_needed()

    # ---- Env helpers ----
    @staticmethod
    def _env(key: str, default: str) -> str:
        return os.environ.get(key, default).strip()

    @staticmethod
    def _env_int(key: str, default: int) -> int:
        raw = os.environ.get(key, str(default)).strip()
        try:
            return int(raw)
        except ValueError:
            warnings.warn(f"Invalid {key}='{raw}', defaulting to {default}.")
            return default

    @staticmethod
    def _env_bool(key: str, default: bool = False) -> bool:
        raw = os.environ.get(key, "1" if default else "0").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    # ---- Shared builders ----
    def _build_embedding_config(self) -> EmbedConfig:
        provider = self._env("EMBEDDING_PROVIDER", "local").lower()
        model = self._env("EMBEDDING_MODEL", "BAAI/bge-m3")
        tokenizer_model = self._env("EMBEDDING_TOKENIZER_MODEL", "") or None
        dimensions = self._env_int("EMBEDDING_DIMENSIONS", 1024)
        base_url = self._env("OLLAMA_BASE_URL", "http://localhost:11434")
        return EmbedConfig(
            provider=provider,
            model=model,
            tokenizer_model=tokenizer_model,
            base_url=base_url,
            dimensions=dimensions,
        )

    def _build_graph_config(self, default_backend: str) -> GraphConfig:
        return GraphConfig(
            backend=self._env("GRAPH_BACKEND", default_backend),
            neo4j_uri=self._env("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=self._env("NEO4J_USER", "neo4j"),
            neo4j_password=self._env("NEO4J_PASSWORD", "password"),
            neo4j_database=self._env("NEO4J_DATABASE", "neo4j"),
        )

    def _offline_mode_enabled(self) -> bool:
        return (
            self._env_bool("GP_OFFLINE")
            or self._env_bool("HF_HUB_OFFLINE")
            or self._env_bool("TRANSFORMERS_OFFLINE")
        )

    def _apply_offline_overrides_if_needed(self):
        """Force internet-free LLM settings when offline flags are enabled."""
        if not self._offline_mode_enabled():
            return

        provider = self._env("OFFLINE_LLM_PROVIDER", "ollama_chat").lower()
        model = self._env("OFFLINE_LLM_MODEL", "qwen2.5:7b")
        base_url = self._env("OFFLINE_LLM_BASE_URL", "http://localhost:11434")
        api_key = self._env("OFFLINE_LLM_API_KEY", "not-needed")

        if provider not in {"ollama_chat", "openai_compatible"}:
            warnings.warn(
                f"Unsupported OFFLINE_LLM_PROVIDER='{provider}', using ollama_chat instead."
            )
            provider = "ollama_chat"

        llm = LLMConfig(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=None if provider == "ollama_chat" else api_key,
            temperature=self.main_llm.temperature,
            max_tokens=self.main_llm.max_tokens,
        )
        self.main_llm = llm
        self.fast_llm = llm

    def _init_dev(self):
        """Development config: Groq if available, else local Ollama."""
        groq_key = self._env("GROQ_API_KEY", "")

        if groq_key:
            self.main_llm = LLMConfig(
                provider="groq",
                model="llama-3.3-70b-versatile",
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1",
            )
            self.fast_llm = LLMConfig(
                provider="groq",
                model="llama-3.1-8b-instant",
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1",
            )
        else:
            # Fallback to local Ollama — pull a small model: ollama pull qwen2.5:7b
            self.main_llm = LLMConfig(
                provider="ollama_chat",
                model="qwen2.5:7b",
                base_url="http://localhost:11434",
            )
            self.fast_llm = LLMConfig(
                provider="ollama_chat",
                model="qwen2.5:7b",
                base_url="http://localhost:11434",
            )

        self.embedding = self._build_embedding_config()
        self.vector_db = VectorDBConfig(
            host=self._env("QDRANT_HOST", "localhost"),
            port=self._env_int("QDRANT_PORT", 6333),
            collection_name=self._env("QDRANT_COLLECTION", "banking_chunks"),
            use_in_memory=self._env_bool("QDRANT_USE_IN_MEMORY", True),
        )
        self.graph = self._build_graph_config(default_backend="networkx")
        self.retrieval = RetrievalConfig()
        self.chunking = ChunkingConfig()

    def _init_prod(self):
        """Production config: fully on-prem vLLM + local everything."""
        self.main_llm = LLMConfig(
            provider="openai_compatible",
            model="Qwen/Qwen2.5-72B-Instruct-AWQ",
            base_url="http://machine1:8000/v1",
            api_key="not-needed",
        )
        self.fast_llm = LLMConfig(
            provider="openai_compatible",
            model="Qwen/Qwen2.5-7B-Instruct",
            base_url="http://machine2:8000/v1",
            api_key="not-needed",
        )

        self.embedding = self._build_embedding_config()
        self.vector_db = VectorDBConfig(
            host=self._env("QDRANT_HOST", "localhost"),
            port=self._env_int("QDRANT_PORT", 6333),
            collection_name=self._env("QDRANT_COLLECTION", "banking_chunks"),
            use_in_memory=self._env_bool("QDRANT_USE_IN_MEMORY", False),
        )
        self.graph = self._build_graph_config(default_backend="neo4j")
        self.retrieval = RetrievalConfig()
        self.chunking = ChunkingConfig()


# ── .env loader (single source of truth) ──
def _load_dotenv():
    """Load .env file if present. Called once before config init."""
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")  # handle quoted values
                os.environ.setdefault(k.strip(), v)


# ── Singleton ──
_config: Optional[Config] = None
_config_lock = threading.Lock()


def get_config(env: Optional[str] = None) -> Config:
    """Get or create the global config (thread-safe)."""
    global _config
    with _config_lock:
        if _config is None or (env and _config.env != env):
            _load_dotenv()
            _config = Config(env=env or os.environ.get("GP_ENV", "dev"))
    return _config
