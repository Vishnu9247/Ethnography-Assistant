"""Project configuration for the ethnographic AI system.

This file keeps all paths and runtime settings in one place so the rest of the
project can stay very small and easy to read.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback for first-run environments
    def load_dotenv() -> None:
        """Fallback no-op when python-dotenv is not installed yet."""
        return None


# Load environment variables from a local .env file if it exists.
load_dotenv()


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
CHROMA_DIR = PROJECT_ROOT / "chroma_store"
PERSONAS_FILE = DATA_DIR / "personas.xlsx"


# Ollama / model settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


# Runtime settings
JSON_RETRY_COUNT = int(os.getenv("JSON_RETRY_COUNT", "3"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "3"))
DETERMINISTIC_MODE = os.getenv("DETERMINISTIC_MODE", "true").lower() == "true"
TEMPERATURE = 0.0 if DETERMINISTIC_MODE else float(os.getenv("TEMPERATURE", "0.3"))
MAX_ETHNOGRAPHY_DOMAINS = int(os.getenv("MAX_ETHNOGRAPHY_DOMAINS", "4"))
MAX_ROWS_FOR_SAMPLE_DATA = int(os.getenv("MAX_ROWS_FOR_SAMPLE_DATA", "3"))


def ensure_directories() -> None:
    """Create runtime directories if they do not already exist."""
    for path in [DATA_DIR, RESULTS_DIR, LOGS_DIR, CHROMA_DIR]:
        path.mkdir(parents=True, exist_ok=True)
