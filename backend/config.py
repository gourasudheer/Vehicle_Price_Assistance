"""
Central configuration for the RAG backend.

All secrets are read from environment variables (see .env.example).
Never hardcode API keys in source files.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # loads variables from a .env file in the working directory, if present

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", BASE_DIR / "chroma_store"))
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# OpenRouter (LLM generation) -- https://openrouter.ai/keys
# ---------------------------------------------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Any OpenRouter chat model works. Good free/cheap defaults for practising:
#   "meta-llama/llama-3.1-8b-instruct:free"
#   "google/gemma-2-9b-it:free"
#   "openai/gpt-4o-mini"  (paid, higher quality)
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")

# Optional but recommended by OpenRouter for attribution/analytics
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://localhost:8501")
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "vehicle-price-rag-chatbot")

# ---------------------------------------------------------------------------
# ChromaDB (vector database) -- runs 100% locally, no account/API key needed
# ---------------------------------------------------------------------------
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "vehicle_prices")

# Local embedding model (downloaded once from HuggingFace, then cached).
# Runs on CPU, free, no API key. ~80MB.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Retrieval / chunking
# ---------------------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "5"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))       # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))  # characters


def require_openrouter_key():
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Create a .env file (see .env.example) "
            "or export it in your shell before running the app."
        )
