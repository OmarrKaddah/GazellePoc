"""
Embedding and vector store indexing.
Embeds chunks using local Hugging Face models and stores in Qdrant.
"""

import re
import warnings
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)

from config import get_config, Config
from src.ingestion.chunker import Chunk


# nomic-embed-text context limit = 8192 tokens.  Conservative safety margin.
_EMBED_TOKEN_LIMIT = 7500


class Embedder:
    """Generates embeddings using the configured provider."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self._local_model = None
        self._tokenizer = None
        self._log_path = Path("data/parsed/embedding_debug.log")

    def _log_embedding_event(self, text: str, token_count: int, vector: list[float]):
        """Append embedding event line including full vector for verification/debugging."""
        try:
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp_utc": f"{datetime.utcnow().isoformat()}Z",
                "provider": self._provider(),
                "model": self.config.embedding.model,
                "tokens": token_count,
                "dim": len(vector),
                "text_sha": text_hash,
                "vector": vector,
            }
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception:
            # Logging must never break embeddings.
            pass

    def _provider(self) -> str:
        provider = (self.config.embedding.provider or "").strip().lower()
        # Backwards compatibility for older config values.
        if provider in {"local", "hf", "huggingface"}:
            return "local"
        return provider

    def _get_local_model(self):
        if self._local_model is not None:
            return self._local_model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "Local embeddings require sentence-transformers. "
                "Install dependencies from requirements.txt."
            ) from e
        self._local_model = SentenceTransformer(self.config.embedding.model)
        return self._local_model

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Exact tokenization requires transformers. "
                "Install dependencies from requirements.txt."
            ) from e

        tokenizer_model = self.config.embedding.tokenizer_model or self.config.embedding.model
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        return self._tokenizer

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the configured local tokenizer."""
        provider = self._provider()
        if provider == "local":
            tokenizer = self._get_tokenizer()
            return len(tokenizer.encode(text, add_special_tokens=False))
        raise ValueError(
            f"Unsupported embedding provider: {self.config.embedding.provider}. "
            "This project uses local Hugging Face embeddings only."
        )

    def _truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """Truncate text to max token budget using the local tokenizer."""
        provider = self._provider()
        if provider == "local":
            tokenizer = self._get_tokenizer()
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) <= max_tokens:
                return text
            truncated_ids = token_ids[:max_tokens]
            return tokenizer.decode(truncated_ids, skip_special_tokens=True)
        raise ValueError(
            f"Unsupported embedding provider: {self.config.embedding.provider}. "
            "This project uses local Hugging Face embeddings only."
        )

    def embed(self, text: str, max_retries: int = 3) -> list[float]:
        """Embed a single text string with retry logic for transient failures."""
        import time

        est_tokens = self._count_tokens(text)
        if est_tokens > _EMBED_TOKEN_LIMIT:
            warnings.warn(
                f"Embedding input too long ({est_tokens} est. tokens, "
                f"{len(text)} chars). Truncating to ~{_EMBED_TOKEN_LIMIT} tokens "
                f"before embedding."
            )
            text = self._truncate_to_token_limit(text, _EMBED_TOKEN_LIMIT)

        provider = self._provider()
        if provider != "local":
            raise ValueError(
                f"Unsupported embedding provider: {self.config.embedding.provider}. "
                "This project uses local Hugging Face embeddings only."
            )

        model = self._get_local_model()
        embedding = model.encode([text], normalize_embeddings=True)[0]
        vector = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
        self._log_embedding_event(text, est_tokens, vector)
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        # Ollama doesn't have native batch — iterate
        return [self.embed(t) for t in texts]


class VectorStore:
    """Qdrant-based vector store with metadata filtering."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        vc = self.config.vector_db

        if vc.use_in_memory:
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(host=vc.host, port=vc.port)

        model_slug = re.sub(r"[^a-zA-Z0-9]+", "_", self.config.embedding.model).strip("_").lower()
        self.collection_name = f"{vc.collection_name}__{model_slug}"
        self._embedder = Embedder(self.config)
        self._embedding_dim: Optional[int] = None
        # Cache: chunk_id -> embedding vector (for graph-discovered chunk scoring)
        self._embedding_cache: dict[str, list[float]] = {}

    def _ensure_collection(self, dim: int):
        """Create collection if it doesn't exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            print(f"Created Qdrant collection '{self.collection_name}' (dim={dim})")

    def index_chunks(self, chunks: list[Chunk], batch_size: int = 32):
        """Embed and index all chunks into Qdrant."""
        if not chunks:
            print("No chunks to index.")
            return

        print(f"Embedding and indexing {len(chunks)} chunks...")

        # Get embedding dimension from first chunk
        first_embedding = self._embedder.embed(chunks[0].content)
        self._embedding_dim = len(first_embedding)
        self._ensure_collection(self._embedding_dim)

        points: list[PointStruct] = []
        embeddings_cache = {0: first_embedding}
        self._embedding_cache[chunks[0].chunk_id] = first_embedding
        #cache when embedding then call for reuse
        for i, chunk in enumerate(chunks):
            if i % 10 == 0:
                print(f"  Embedding chunk {i + 1}/{len(chunks)}...")

            if i in embeddings_cache:
                embedding = embeddings_cache[i]
            else:
                embedding = self._embedder.embed(chunk.content)

            # Cache for later lookup by chunk_id
            self._embedding_cache[chunk.chunk_id] = embedding

            payload = {
                "chunk_id": chunk.chunk_id,
                "doc_name": chunk.doc_name,
                "doc_path": chunk.doc_path,
                "element_type": chunk.element_type,
                "content": chunk.content,
                "section_path": chunk.section_path,
                "section_path_str": chunk.metadata.get("section_path_str", ""),
                "source_element_ids": chunk.source_element_ids,
                "language": chunk.language,
                "token_estimate": chunk.token_estimate,
                "access_level": chunk.metadata.get("access_level", 1),
            }
            
            #points heya fundamnetal unit of storage in qdrant vectordb
            #points contain an id, the embedding vector, and a payload (metadata) 
            #which can be used for filtering and retrieval
            points.append(PointStruct(
                id=i,
                vector=embedding,
                payload=payload,
            ))

            # Batches points to avoid large payloads in a single request
            #matters when qdranrt is not local
            if len(points) >= batch_size:
                self.client.upsert(collection_name=self.collection_name, points=points)
                points = []

        # Flush remaining
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)

        print(f"Indexed {len(chunks)} chunks into '{self.collection_name}'")

    def search(
        self,
        query: str,
        top_k: int = 10,
        doc_filter: Optional[str] = None,
        element_type_filter: Optional[str] = None,
        max_access_level: Optional[int] = None,
    ) -> list[dict]:
        """
        Search for similar chunks.
        Returns list of dicts with 'score', 'content', and all payload fields.

        Args:
            max_access_level: If set, only return chunks with access_level <= this value.
                              1 = Public only, 2 = Public+Confidential, 3 = all.
        """
        query_embedding = self._embedder.embed(query)

        # Build filter conditions
        conditions = []
        if doc_filter:
            conditions.append(FieldCondition(key="doc_name", match=MatchValue(value=doc_filter)))
        if element_type_filter:
            conditions.append(FieldCondition(key="element_type", match=MatchValue(value=element_type_filter)))
        if max_access_level is not None:
            conditions.append(FieldCondition(key="access_level", range=Range(lte=max_access_level)))

        search_filter = Filter(must=conditions) if conditions else None

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=search_filter,
        )

        return [
            {
                "score": hit.score,
                **hit.payload,
            }
            for hit in results.points
        ]

    def get_collection_info(self) -> dict:
        """Get collection stats."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
        }

    def get_cached_embedding(self, chunk_id: str) -> Optional[list[float]]:
        """Return cached embedding for a chunk_id, or None if not cached."""
        return self._embedding_cache.get(chunk_id)

    def compute_similarity(self, query: str, chunk_id: str) -> float:
        """Compute cosine similarity between query and a cached chunk embedding."""
        import numpy as np
        chunk_emb = self._embedding_cache.get(chunk_id)
        if chunk_emb is None:
            return 0.0
        query_emb = self._embedder.embed(query)
        a = np.array(query_emb, dtype=np.float32)
        b = np.array(chunk_emb, dtype=np.float32)
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / norm) if norm > 0 else 0.0
