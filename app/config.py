"""
Centralized Application Configuration

All API keys, paths, model names, and constants in one place.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ─────────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found in environment variables. "
        "Please add it to your .env file."
    )

# ── Model Configuration ─────────────────────────────────────────────────────
LLM_MODEL_NAME = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.5
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ── Paths ────────────────────────────────────────────────────────────────────
PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "database")
PROMPT_TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "prompts", "templates")
BERT_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "legal-bert-clause-classifier")

# ── Text Splitting ───────────────────────────────────────────────────────────
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ── Retriever ────────────────────────────────────────────────────────────────
RETRIEVER_SEARCH_TYPE = "mmr"
RETRIEVER_TOP_K = 3
