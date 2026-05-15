"""
RAG (Retrieval-Augmented Generation) module with ChromaDB integration.
Supports multiple embeddings strategies and semantic search.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json
from src.common import get_logger, RAGError


logger = get_logger(__name__)


class EmbeddingsProvider(ABC):
    """Abstract base class for embeddings providers."""

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        """
        Embed text to vector.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts.

        Args:
            texts: List of texts

        Returns:
            List of embedding vectors
        """
        pass


class MockEmbeddings(EmbeddingsProvider):
    """Mock embeddings for testing (free, no API calls)."""

    def embed(self, text: str) -> List[float]:
        """Generate mock embedding."""
        import hashlib

        hash_value = hashlib.md5(text.encode()).hexdigest()
        return [float(int(hash_value[i : i + 2], 16)) / 256 for i in range(0, 32, 2)]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for batch."""
        return [self.embed(text) for text in texts]


class HuggingFaceEmbeddings(EmbeddingsProvider):
    """HuggingFace embeddings (free, local)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize HuggingFace embeddings."""
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name)
            logger.info(f"Initialized HuggingFace embeddings with model: {model_name}")
        except ImportError:
            raise RAGError(
                "sentence-transformers not found. Install with: pip install sentence-transformers",
                retryable=False,
            )

    def embed(self, text: str) -> List[float]:
        """Embed text using HuggingFace."""
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            raise RAGError(f"HuggingFace embedding failed: {str(e)}")

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed batch using HuggingFace."""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            raise RAGError(f"HuggingFace batch embedding failed: {str(e)}")


class OpenAIEmbeddings(EmbeddingsProvider):
    """OpenAI embeddings API."""

    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        """Initialize OpenAI embeddings."""
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=api_key)
            self.model_name = model_name
            logger.info(f"Initialized OpenAI embeddings with model: {model_name}")
        except ImportError:
            raise RAGError(
                "openai not found. Install with: pip install openai",
                retryable=False,
            )

    def embed(self, text: str) -> List[float]:
        """Embed text using OpenAI."""
        try:
            response = self.client.embeddings.create(input=text, model=self.model_name)
            return response.data[0].embedding
        except Exception as e:
            raise RAGError(f"OpenAI embedding failed: {str(e)}", retryable=True)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed batch using OpenAI."""
        try:
            response = self.client.embeddings.create(input=texts, model=self.model_name)
            return [item.embedding for item in response.data]
        except Exception as e:
            raise RAGError(f"OpenAI batch embedding failed: {str(e)}", retryable=True)


class RAGManager:
    """Manager for RAG operations with ChromaDB."""

    def __init__(
        self,
        embeddings_provider: EmbeddingsProvider,
        collection_prefix: str = "travel_ai",
        db_path: Optional[str] = None,
    ):
        """
        Initialize RAG manager.

        Args:
            embeddings_provider: Embeddings provider instance
            collection_prefix: Prefix for collection names
            db_path: Path to ChromaDB persistent storage
        """
        self.embeddings_provider = embeddings_provider
        self.collection_prefix = collection_prefix

        try:
            import chromadb
            from pathlib import Path

            # Create persistent storage path if provided
            if db_path:
                Path(db_path).mkdir(parents=True, exist_ok=True)
                try:
                    # Try new Chroma client API (0.4.x+)
                    self.client = chromadb.PersistentClient(path=db_path)
                    logger.info(f"Initialized ChromaDB client with persistent storage at: {db_path}")
                except Exception as e:
                    # Fallback to in-memory if persistent fails
                    logger.warning(f"Failed to initialize persistent Chroma: {str(e)}. Using in-memory instead.")
                    self.client = chromadb.EphemeralClient()
                    logger.info("Initialized ChromaDB client (in-memory fallback)")
            else:
                # Use ephemeral client for in-memory storage
                self.client = chromadb.EphemeralClient()
                logger.info("Initialized ChromaDB client (in-memory)")
        except ImportError:
            raise RAGError(
                "chromadb not found. Install with: pip install chromadb",
                retryable=False,
            )
        except Exception as e:
            logger.warning(f"RAG initialization error: {str(e)}. Continuing without RAG.")
            self.client = None

        # Initialize collections
        self._initialize_collections()

    def _initialize_collections(self) -> None:
        """Initialize all RAG collections."""
        self.collections = {}

        # If client failed to initialize, skip collection initialization
        if not self.client:
            logger.warning("RAG client not initialized. Skipping collection initialization.")
            return

        collection_names = [
            "user_profiles",
            "query_history",
            "constraints",
            "patterns",
            "travel_tips",
        ]

        for name in collection_names:
            full_name = f"{self.collection_prefix}_{name}"
            try:
                self.collections[name] = self.client.get_or_create_collection(
                    name=full_name,
                    metadata={"hnsw:space": "cosine"},
                )
                logger.info(f"Initialized collection: {full_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize collection {full_name}: {str(e)}")

    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """
        Add documents to collection.

        Args:
            collection_name: Collection name
            documents: List of document texts
            metadatas: Optional metadata for each document
            ids: Optional document IDs
        """
        try:
            if collection_name not in self.collections:
                raise ValueError(f"Unknown collection: {collection_name}")

            embeddings = self.embeddings_provider.embed_batch(documents)

            self.collections[collection_name].add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas or [{} for _ in documents],
                ids=ids or [f"doc_{i}" for i in range(len(documents))],
            )
            logger.info(f"Added {len(documents)} documents to {collection_name}")
        except Exception as e:
            raise RAGError(f"Failed to add documents: {str(e)}")

    def search(
        self,
        collection_name: str,
        query: str,
        n_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Search collection.

        Args:
            collection_name: Collection name
            query: Search query
            n_results: Number of results

        Returns:
            Search results with documents and distances
        """
        try:
            if collection_name not in self.collections:
                raise ValueError(f"Unknown collection: {collection_name}")

            query_embedding = self.embeddings_provider.embed(query)
            results = self.collections[collection_name].query(
                query_embeddings=[query_embedding],
                n_results=n_results,
            )

            return {
                "documents": results["documents"][0] if results["documents"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
            }
        except Exception as e:
            raise RAGError(f"Search failed: {str(e)}")

    def delete_collection(self, collection_name: str) -> None:
        """
        Delete collection.

        Args:
            collection_name: Collection name
        """
        try:
            if collection_name in self.collections:
                full_name = f"{self.collection_prefix}_{collection_name}"
                self.client.delete_collection(name=full_name)
                del self.collections[collection_name]
                logger.info(f"Deleted collection: {full_name}")
        except Exception as e:
            raise RAGError(f"Failed to delete collection: {str(e)}")


def create_embeddings_provider(provider_type: str, **kwargs) -> EmbeddingsProvider:
    """
    Factory function to create embeddings provider.

    Args:
        provider_type: Type of embeddings ("mock", "huggingface", "openai")
        **kwargs: Provider-specific arguments (api_key for openai only, model_name for huggingface)

    Returns:
        Embeddings provider instance
    """
    try:
        if provider_type == "mock":
            return MockEmbeddings()
        elif provider_type == "huggingface":
            # HuggingFace embeddings don't accept api_key parameter
            # But they do accept model_name and can use HF_TOKEN from environment
            valid_kwargs = {k: v for k, v in kwargs.items() if k not in ["api_key"]}
            return HuggingFaceEmbeddings(**valid_kwargs)
        elif provider_type == "openai":
            if "api_key" not in kwargs or not kwargs["api_key"]:
                logger.warning("OpenAI API key not provided, falling back to mock embeddings")
                return MockEmbeddings()
            return OpenAIEmbeddings(**kwargs)
        else:
            logger.warning(f"Unknown embeddings provider: {provider_type}, using mock")
            return MockEmbeddings()
    except Exception as e:
        logger.error(f"Failed to create embeddings provider {provider_type}: {e}")
        logger.info("Falling back to mock embeddings")
        return MockEmbeddings()
