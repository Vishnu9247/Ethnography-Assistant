"""Small Ollama helper functions used by all agents."""

from __future__ import annotations

import json
from typing import Any, Dict

from langchain_ollama import ChatOllama

from config import JSON_RETRY_COUNT, OLLAMA_BASE_URL, OLLAMA_MODEL, TEMPERATURE


def _build_llm(json_mode: bool = False) -> ChatOllama:
    """Create a ChatOllama client with stable defaults."""
    kwargs: Dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "base_url": OLLAMA_BASE_URL,
        "temperature": TEMPERATURE,
    }
    if json_mode:
        kwargs["format"] = "json"
    return ChatOllama(**kwargs)


def ask_llm(prompt: str) -> str:
    """Send a plain-text prompt to Ollama and return text."""
    llm = _build_llm(json_mode=False)
    response = llm.invoke(prompt)
    return response.content.strip()


def structured_json(prompt: str) -> Dict[str, Any]:
    """Ask the model for JSON with retries if parsing fails."""
    llm = _build_llm(json_mode=True)
    last_error = None

    for attempt in range(1, JSON_RETRY_COUNT + 1):
        response = llm.invoke(prompt)
        raw_text = response.content.strip()
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as error:
            last_error = error
            repair_prompt = (
                "Return valid JSON only. Fix this content and keep the same meaning:\n"
                f"{raw_text}"
            )
            response = llm.invoke(repair_prompt)
            try:
                return json.loads(response.content.strip())
            except json.JSONDecodeError as second_error:
                last_error = second_error
                if attempt == JSON_RETRY_COUNT:
                    break

    raise ValueError(f"Could not parse LLM JSON response after retries: {last_error}")
