"""
Embedding and vector store indexing.
Embeds chunks using Ollama (dev) or local model (prod) and stores in Qdrant.
"""

import json
from pathlib import Path
from typing import Optional

import ollama as ollama_client
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from config import get_config, Config
from src.ingestion.chunker import Chunk


class Embedder:
    """Generates embeddings using the configured provider."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()

    def embed(self, text: str, max_retries: int = 3) -> list[float]:
        """Embed a single text string with retry logic for transient failures."""
        import time
        if self.config.embedding.provider == "ollama":
            for attempt in range(max_retries):
                try:
                    response = ollama_client.embeddings(
                        model=self.config.embedding.model,
                        prompt=text,
                    )
                    return response["embedding"]
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = 2 ** attempt
                        print(f"  ⚠ Ollama embed retry {attempt+1}/{max_retries} after {wait}s: {e}")
                        time.sleep(wait)
                    else:
                        raise
        else:
            raise ValueError(f"Unsupported embedding provider: {self.config.embedding.provider}")

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

        self.collection_name = vc.collection_name
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
    ) -> list[dict]:
        """
        Search for similar chunks.
        Returns list of dicts with 'score', 'content', and all payload fields.
        """
        query_embedding = self._embedder.embed(query)

        # Build filter conditions
        conditions = []
        if doc_filter:
            conditions.append(FieldCondition(key="doc_name", match=MatchValue(value=doc_filter)))
        if element_type_filter:
            conditions.append(FieldCondition(key="element_type", match=MatchValue(value=element_type_filter)))

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
