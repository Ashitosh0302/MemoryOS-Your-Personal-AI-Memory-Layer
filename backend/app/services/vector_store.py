"""
Vector store service for MemoryOS using ChromaDB.

Provides persistent storage and similarity search for document text chunks
and their corresponding vector embeddings.
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# Default persistent directory: backend/chroma_db
DEFAULT_DB_DIR = Path(__file__).resolve().parents[2] / "chroma_db"
DEFAULT_COLLECTION_NAME = "memory_documents"


class VectorStoreService:
    """
    Service wrapper for persistent ChromaDB storage.
    """

    def __init__(
        self,
        db_dir: Optional[Path | str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ):
        self.db_dir = Path(db_dir) if db_dir else DEFAULT_DB_DIR
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        logger.info(
            "Initializing ChromaDB client at: %s (collection: %s)",
            self.db_dir,
            self.collection_name,
        )

        self.client = chromadb.PersistentClient(
            path=str(self.db_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Add text chunks and vector embeddings to the ChromaDB collection.

        Parameters
        ----------
        chunks : List[str]
            List of chunk text strings.
        embeddings : List[List[float]]
            List of embedding vectors corresponding to chunks.
        metadatas : Optional[List[Dict[str, Any]]]
            Optional metadata dict for each chunk.
        ids : Optional[List[str]]
            Optional unique ID strings for each chunk. Generated if not provided.

        Returns
        -------
        List[str]
            List of chunk IDs added.
        """
        if not chunks:
            return []

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: got {len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        if metadatas is not None and len(metadatas) != len(chunks):
            raise ValueError(
                f"Mismatch: got {len(chunks)} chunks but {len(metadatas)} metadatas."
            )

        if ids is None:
            chunk_ids = [str(uuid.uuid4()) for _ in chunks]
        else:
            if len(ids) != len(chunks):
                raise ValueError(
                    f"Mismatch: got {len(chunks)} chunks but {len(ids)} ids."
                )
            chunk_ids = ids

        clean_metadatas = None
        if metadatas is not None:
            processed = []
            for meta in metadatas:
                clean_meta = {}
                for k, v in meta.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_meta[k] = v
                    else:
                        clean_meta[k] = str(v)
                processed.append(clean_meta)
            
            # If any metadata dictionary is non-empty, pass processed list, otherwise None
            if any(len(m) > 0 for m in processed):
                clean_metadatas = processed

        self.collection.add(
            ids=chunk_ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=clean_metadatas,
        )

        logger.info("Successfully added %d chunks to ChromaDB", len(chunks))
        return chunk_ids

    def get_count(self) -> int:
        """Return total number of chunks stored in the collection."""
        return self.collection.count()

    def query_similar(
        self,
        query_embedding: List[float],
        n_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Query collection for chunks similar to the query embedding vector.

        Parameters
        ----------
        query_embedding : List[float]
            The vector embedding of the query.
        n_results : int, default=5
            Maximum number of top results to return.

        Returns
        -------
        List[Dict[str, Any]]
            List of result dicts containing:
            - 'id': str
            - 'text': str
            - 'metadata': dict
            - 'distance': float
        """
        if self.get_count() == 0:
            return []

        if not query_embedding:
            return []

        actual_n = min(n_results, self.get_count())
        if actual_n <= 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_n,
            include=["documents", "metadatas", "distances"],
        )

        output: List[Dict[str, Any]] = []

        ids_list = results.get("ids", [[]])[0]
        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        dists_list = results.get("distances", [[]])[0]

        for i in range(len(ids_list)):
            output.append(
                {
                    "id": ids_list[i],
                    "text": docs_list[i] if docs_list else "",
                    "metadata": metas_list[i] if metas_list else {},
                    "distance": dists_list[i] if dists_list else 0.0,
                }
            )

        return output


# ---------------------------------------------------------------------------
# Module-level convenience instance / functions
# ---------------------------------------------------------------------------

_default_store: Optional[VectorStoreService] = None


def get_vector_store(
    db_dir: Optional[Path | str] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
) -> VectorStoreService:
    """Return a VectorStoreService instance."""
    global _default_store
    if db_dir is not None or collection_name != DEFAULT_COLLECTION_NAME:
        return VectorStoreService(db_dir=db_dir, collection_name=collection_name)

    if _default_store is None:
        _default_store = VectorStoreService()

    return _default_store
