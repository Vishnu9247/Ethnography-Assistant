"""Agent 3: memory retrieval, pattern detection, and recommendations."""

from __future__ import annotations

from typing import Any, Dict

from config import DEFAULT_TOP_K
from memory.vector_store import search
from utils.llm import structured_json
from utils.logger import get_logger
from utils.prompts import build_analysis_prompt


LOGGER = get_logger("agent3")


def run_agent3(
    persona_data: Dict[str, str],
    intake_result: Dict[str, Any],
    ethnography_result: Dict[str, Any],
    session_id: str,
) -> Dict[str, Any]:
    """Run the analysis step using stored domain summaries."""
    LOGGER.info("Running Agent 3 for session %s", session_id)

    retrieved_items = search(
        session_id=session_id,
        query=intake_result["final_problem_statement"],
        top_k=DEFAULT_TOP_K,
    )

    result = structured_json(
        build_analysis_prompt(
            problem_statement=intake_result["final_problem_statement"],
            person_details=persona_data["Person Details"],
            retrieved_items=retrieved_items,
            expected_solution=persona_data["Ethnographic Solution"],
            domain_summaries=ethnography_result["domain_summaries"],
        )
    )

    result["retrieved_items"] = retrieved_items
    return result
