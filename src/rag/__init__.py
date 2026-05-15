"""
RAG (Retrieval-Augmented Generation) module: ChromaDB integration with multiple embeddings strategies.
"""

from .manager import (
    EmbeddingsProvider,
    MockEmbeddings,
    HuggingFaceEmbeddings,
    OpenAIEmbeddings,
    RAGManager,
    create_embeddings_provider,
)

__all__ = [
    "EmbeddingsProvider",
    "MockEmbeddings",
    "HuggingFaceEmbeddings",
    "OpenAIEmbeddings",
    "RAGManager",
    "create_embeddings_provider",
]
