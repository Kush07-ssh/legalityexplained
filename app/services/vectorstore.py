"""
Vector Store Service

Manages ChromaDB vector store creation, persistence, and retrieval.
"""

import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    PERSIST_DIR,
    EMBEDDING_MODEL_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVER_SEARCH_TYPE,
    RETRIEVER_TOP_K,
)

# Singleton embeddings instance (loaded once)
_embeddings = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Get or create the embeddings model (singleton)."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return _embeddings


def create_vectorstore(pages: List[Document]) -> Chroma:
    """
    Create a new vector store from document pages.

    Splits pages into chunks, embeds them, and persists to disk.
    If a vector store already exists, loads it instead.
    """
    embeddings = _get_embeddings()

    if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        chunks = text_splitter.split_documents(pages)
        vectorstore = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=PERSIST_DIR,
        )
    else:
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
        )

    return vectorstore


def get_vectorstore() -> Chroma:
    """Load the existing persisted vector store."""
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=_get_embeddings(),
    )


def get_retriever(vectorstore: Optional[Chroma] = None):
    """Get a retriever from the vector store."""
    if vectorstore is None:
        vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_type=RETRIEVER_SEARCH_TYPE,
        search_kwargs={"k": RETRIEVER_TOP_K},
    )
