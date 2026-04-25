"""Agent 1: intake and problem structuring."""

from __future__ import annotations

from typing import Any, Dict, List

from utils.llm import structured_json
from utils.prompts import build_intake_structuring_prompt
from utils.logger import get_logger


LOGGER = get_logger("agent1")


def _detect_age_hint(person_details: str) -> str:
    """Try to pull a simple age hint from details without complex parsing."""
    if "year-old" in person_details:
        left_side = person_details.split("year-old", 1)[0].strip()
        tokens = left_side.split()
        if tokens:
            return tokens[-1]
    return "unknown"


def run_agent1(persona_agent: Any, persona_data: Dict[str, str], session_id: str) -> Dict[str, Any]:
    """Run the intake flow with a persona simulator."""
    LOGGER.info("Running Agent 1 for session %s", session_id)

    conversation: List[Dict[str, str]] = []

    # Ask simple baseline questions so the intake is explicit and easy to inspect.
    questions = [
        "Hello. I would like to understand your situation better. What is your name?",
        "How old are you, and what part of your life feels most affected right now?",
        "What problem is bothering you the most at the moment?",
        "Can you give me one concrete recent example that shows this problem clearly?",
    ]

    for question in questions:
        answer = persona_agent.respond(question)
        conversation.append({"question": question, "answer": answer})

    transcript_text = "\n".join(
        f"Q: {item['question']}\nA: {item['answer']}" for item in conversation
    )

    result = structured_json(
        build_intake_structuring_prompt(
            persona_name=persona_data["Name"],
            known_problem=persona_data["Problem"],
            known_details=persona_data["Person Details"],
            transcript=transcript_text,
            age_hint=_detect_age_hint(persona_data["Person Details"]),
        )
    )

    result["conversation"] = conversation
    result["session_id"] = session_id
    return result
