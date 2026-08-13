"""
Document Loader Service

Handles loading uploaded files (PDF/TXT), extracting text content,
and preprocessing it for analysis.
"""

import os
import re
import tempfile
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_files_from_uploads(uploaded_files: list) -> List[Document]:
    """
    Load content from a list of Streamlit uploaded files (PDFs or TXTs).

    Each file is written to a temporary location for the loader to read,
    then the temp file is cleaned up.
    """
    all_docs = []

    for file in uploaded_files:
        suffix = Path(file.name).suffix.lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        try:
            if suffix == ".txt":
                loader = TextLoader(tmp_path, encoding="utf-8")
                all_docs.extend(loader.load())
            elif suffix == ".pdf":
                loader = PyPDFLoader(tmp_path)
                all_docs.extend(loader.load())
        finally:
            os.remove(tmp_path)

    return all_docs


def preprocess_text(text: str) -> str:
    """
    Clean and normalize legal document text.

    Performs:
    - Lowercasing
    - Page number removal
    - Whitespace consolidation
    - Section symbol normalization
    - Minimal special character removal (preserves important legal punctuation)
    """
    text = text.lower()
    text = re.sub(r'page\s*\d+\s*(of\s*\d+)?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace("§", "section")
    # Minimal removal to preserve important symbols in contracts
    text = re.sub(r'[^a-zA-Z0-9.,;:!?()\-\'\"\s]', '', text)
    return text.strip()


def combine_pages(pages: List[Document]) -> str:
    """Combine all page contents into a single preprocessed string."""
    return "\n".join([preprocess_text(page.page_content) for page in pages])
