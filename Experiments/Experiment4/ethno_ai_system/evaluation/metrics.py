"""Lightweight evaluation metrics for the research prototype."""

from __future__ import annotations

import re
from typing import Dict, List


def _normalize_words(text: str) -> List[str]:
    """Turn text into simple lowercase word tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def word_overlap_score(text_a: str, text_b: str) -> float:
    """Return a simple overlap score between 0 and 1."""
    words_a = set(_normalize_words(text_a))
    words_b = set(_normalize_words(text_b))
    if not words_a or not words_b:
        return 0.0
    return round(len(words_a & words_b) / len(words_a | words_b), 3)


def score_agent1(result: Dict, persona_data: Dict) -> Dict[str, float]:
    """Score Agent 1 output."""
    statement = result["final_problem_statement"]
    completeness = min(len(_normalize_words(statement)) / 30.0, 1.0)
    clarification_turns = max(len(result.get("conversation", [])) - 2, 0)
    accuracy = word_overlap_score(statement, persona_data["Problem"])
    return {
        "agent1_problem_completeness": round(completeness, 3),
        "agent1_clarification_turns": float(clarification_turns),
        "agent1_accuracy_vs_excel": accuracy,
    }


def score_agent2(result: Dict, persona_data: Dict) -> Dict[str, float]:
    """Score Agent 2 output."""
    selected_count = len(result["selected_domains"])
    domain_coverage = round(min(selected_count / 4.0, 1.0), 3)
    redundant_questions = float(result["redundant_questions"])

    all_summary_text = " ".join(result["domain_summaries"].values())
    depth_score = round(min(len(_normalize_words(all_summary_text)) / 80.0, 1.0), 3)
    relevance_score = word_overlap_score(all_summary_text, persona_data["Person Details"])

    return {
        "agent2_domain_coverage": domain_coverage,
        "agent2_redundant_questions": redundant_questions,
        "agent2_depth_score": depth_score,
        "agent2_relevance_score": relevance_score,
    }


def score_agent3(result: Dict, persona_data: Dict) -> Dict[str, float]:
    """Score Agent 3 output."""
    recommendation_text = " ".join(
        item["recommendation"] for item in result.get("recommendations", [])
    )
    pattern_text = " ".join(result.get("patterns", []))
    root_cause_text = " ".join(result.get("root_causes", []))

    recommendation_relevance = word_overlap_score(
        recommendation_text, persona_data["Ethnographic Solution"]
    )
    solution_similarity = word_overlap_score(
        recommendation_text + " " + pattern_text,
        persona_data["Ethnographic Solution"],
    )
    explainability = round(min(len(_normalize_words(root_cause_text)) / 25.0, 1.0), 3)

    return {
        "agent3_recommendation_relevance": recommendation_relevance,
        "agent3_similarity_vs_expected_solution": solution_similarity,
        "agent3_explainability_score": explainability,
    }


def score_full_system(
    agent1_scores: Dict[str, float],
    agent2_scores: Dict[str, float],
    agent3_scores: Dict[str, float],
) -> Dict[str, float]:
    """Create a simple combined score for the full system."""
    numeric_values = [
        value
        for group in [agent1_scores, agent2_scores, agent3_scores]
        for value in group.values()
        if isinstance(value, (int, float))
    ]
    average = round(sum(numeric_values) / len(numeric_values), 3) if numeric_values else 0.0
    return {"full_system_average_score": average}
