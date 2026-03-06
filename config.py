"""
Configuration for the Banking Knowledge AI System.
Switch between dev (Groq + local Ollama) and prod (fully on-prem vLLM).
"""

import os
from dataclasses import dataclass, field
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
        elif self.env == "prod":
            self._init_prod()

    def _init_dev(self):
        """Development config: Groq APIs + local Ollama embeddings.
        Falls back to local Ollama LLM if no Groq key is set."""
        groq_key = os.environ.get("GROQ_API_KEY", "")

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
        self.embedding = EmbedConfig(
            provider="ollama",
            model="nomic-embed-text",
        )
        self.vector_db = VectorDBConfig(use_in_memory=True)
        self.graph = GraphConfig(backend="networkx")
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
        self.embedding = EmbedConfig(
            provider="local",
            model="BAAI/bge-m3",
            dimensions=1024,
        )
        self.vector_db = VectorDBConfig(use_in_memory=False)
        self.graph = GraphConfig(backend="neo4j")
        self.retrieval = RetrievalConfig()
        self.chunking = ChunkingConfig()


# ── Singleton ──
_config: Optional[Config] = None


def get_config(env: Optional[str] = None) -> Config:
    """Get or create the global config."""
    global _config
    if _config is None or (env and _config.env != env):
        _config = Config(env=env or os.environ.get("GP_ENV", "dev"))
    return _config
