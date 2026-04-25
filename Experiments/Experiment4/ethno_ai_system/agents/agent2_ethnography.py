"""Agent 2: ethnographic questioning across life domains."""

from __future__ import annotations

from typing import Any, Dict, List

from config import MAX_ETHNOGRAPHY_DOMAINS
from memory.vector_store import add_document
from utils.llm import structured_json
from utils.logger import get_logger
from utils.prompts import (
    build_domain_planning_prompt,
    build_domain_question_prompt,
    build_domain_summary_prompt,
)


LOGGER = get_logger("agent2")


def _question_seen(previous_questions: List[str], candidate: str) -> bool:
    """Simple duplicate check so the agent avoids repeating itself."""
    candidate_words = set(candidate.lower().split())
    for old_question in previous_questions:
        old_words = set(old_question.lower().split())
        if not candidate_words:
            continue
        overlap_ratio = len(candidate_words & old_words) / len(candidate_words)
        if overlap_ratio > 0.7:
            return True
    return False


def run_agent2(
    persona_agent: Any,
    persona_data: Dict[str, str],
    intake_result: Dict[str, Any],
    session_id: str,
) -> Dict[str, Any]:
    """Run adaptive ethnographic questioning for the most relevant domains."""
    LOGGER.info("Running Agent 2 for session %s", session_id)

    plan = structured_json(
        build_domain_planning_prompt(
            problem_statement=intake_result["final_problem_statement"],
            person_details=persona_data["Person Details"],
            max_domains=MAX_ETHNOGRAPHY_DOMAINS,
        )
    )

    selected_domains = plan.get("selected_domains", [])[:MAX_ETHNOGRAPHY_DOMAINS]
    all_questions: List[str] = []
    domain_summaries: Dict[str, str] = {}
    domain_conversations: Dict[str, List[Dict[str, str]]] = {}
    redundant_questions = 0

    for domain in selected_domains:
        domain_key = domain.lower().replace(" ", "_")
        domain_conversations[domain_key] = []

        for turn_number in [1, 2]:
            question_payload = structured_json(
                build_domain_question_prompt(
                    domain=domain,
                    turn_number=turn_number,
                    problem_statement=intake_result["final_problem_statement"],
                    prior_memory=domain_conversations[domain_key],
                )
            )
            question = question_payload["question"]

            if _question_seen(all_questions, question):
                redundant_questions += 1
                continue

            answer = persona_agent.respond(question)
            exchange = {"question": question, "answer": answer}
            domain_conversations[domain_key].append(exchange)
            all_questions.append(question)

        summary_payload = structured_json(
            build_domain_summary_prompt(
                domain=domain,
                problem_statement=intake_result["final_problem_statement"],
                conversation=domain_conversations[domain_key],
            )
        )
        summary_text = summary_payload["summary"]
        domain_summaries[domain_key] = summary_text
        add_document(session_id=session_id, category=domain_key, text=summary_text)

    return {
        "selected_domains": selected_domains,
        "domain_conversations": domain_conversations,
        "domain_summaries": domain_summaries,
        "redundant_questions": redundant_questions,
    }
