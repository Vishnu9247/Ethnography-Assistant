"""Shared logger setup for the project."""

from __future__ import annotations

import logging
from pathlib import Path

from config import LOGS_DIR, ensure_directories


def get_logger(name: str) -> logging.Logger:
    """Create or return a logger that writes both to file and console."""
    ensure_directories()

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(LOGS_DIR / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
