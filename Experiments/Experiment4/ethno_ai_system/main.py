"""Main entry point for running the full ethnographic agent system."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.agent1_intake import run_agent1
from agents.agent2_ethnography import run_agent2
from agents.agent3_analysis import run_agent3
from config import ensure_directories
from memory.vector_store import delete_session, init_store
from simulator.persona_agent import PersonaAgent
from utils.excel_loader import get_persona_by_row, validate_dataset_exists
from utils.logger import get_logger


class SessionState(TypedDict, total=False):
    """Shared state passed across the LangGraph workflow."""

    session_id: str
    persona_row: int
    persona_data: Dict[str, Any]
    persona_agent: PersonaAgent
    intake_result: Dict[str, Any]
    ethnography_result: Dict[str, Any]
    analysis_result: Dict[str, Any]
    scores: Dict[str, Any]


LOGGER = get_logger("main")


def agent1_node(state: SessionState) -> SessionState:
    """Run intake agent and update graph state."""
    state["intake_result"] = run_agent1(
        persona_agent=state["persona_agent"],
        persona_data=state["persona_data"],
        session_id=state["session_id"],
    )
    return state


def agent2_node(state: SessionState) -> SessionState:
    """Run ethnography agent and update graph state."""
    state["ethnography_result"] = run_agent2(
        persona_agent=state["persona_agent"],
        persona_data=state["persona_data"],
        intake_result=state["intake_result"],
        session_id=state["session_id"],
    )
    return state


def agent3_node(state: SessionState) -> SessionState:
    """Run analysis agent and update graph state."""
    state["analysis_result"] = run_agent3(
        persona_data=state["persona_data"],
        intake_result=state["intake_result"],
        ethnography_result=state["ethnography_result"],
        session_id=state["session_id"],
    )
    return state


def build_graph():
    """Create the LangGraph workflow."""
    graph = StateGraph(SessionState)
    graph.add_node("agent1", agent1_node)
    graph.add_node("agent2", agent2_node)
    graph.add_node("agent3", agent3_node)
    graph.add_edge(START, "agent1")
    graph.add_edge("agent1", "agent2")
    graph.add_edge("agent2", "agent3")
    graph.add_edge("agent3", END)
    return graph.compile()


def run_session(row_number: int, cleanup: bool = True) -> SessionState:
    """Run the full pipeline for one persona row."""
    ensure_directories()
    validate_dataset_exists()
    init_store()

    persona_data = get_persona_by_row(row_number)
    persona_agent = PersonaAgent.from_row(row_number)
    session_id = f"row_{row_number}"

    LOGGER.info("Starting session for row %s (%s)", row_number, persona_data["Name"])

    graph = build_graph()
    state: SessionState = {
        "session_id": session_id,
        "persona_row": row_number,
        "persona_data": persona_data,
        "persona_agent": persona_agent,
    }

    final_state = graph.invoke(state)

    if cleanup:
        delete_session(session_id)

    return final_state


def format_session_report(session_state: SessionState) -> str:
    """Create a terminal-friendly session report."""
    persona_name = session_state["persona_data"]["Name"]
    intake_result = session_state["intake_result"]
    ethnography_result = session_state["ethnography_result"]
    analysis_result = session_state["analysis_result"]

    domain_lines = []
    for domain, summary in ethnography_result["domain_summaries"].items():
        title = domain.replace("_", " ").title()
        domain_lines.append(f"{title}:\n{summary}")

    recommendation_lines = []
    for item in analysis_result["recommendations"]:
        recommendation_lines.append(
            f"- Rank {item['rank']}: {item['recommendation']} ({item['reason']})"
        )

    report = "\n".join(
        [
            "=" * 50,
            "SESSION REPORT",
            "=" * 50,
            f"Persona: {persona_name}",
            "",
            "Problem Statement:",
            intake_result["final_problem_statement"],
            "",
            "Agent 2 Findings:",
            "\n\n".join(domain_lines) if domain_lines else "No domains collected.",
            "",
            "Patterns:",
            "\n".join(f"- {item}" for item in analysis_result["patterns"]),
            "",
            "Root Causes:",
            "\n".join(f"- {item}" for item in analysis_result["root_causes"]),
            "",
            "Recommendations:",
            "\n".join(recommendation_lines),
            "",
            "Scores:",
            str(session_state.get("scores", {})),
            "=" * 50,
        ]
    )
    return report


def parse_args() -> argparse.Namespace:
    """Parse terminal arguments."""
    parser = argparse.ArgumentParser(description="Run the Ethnographic Multi-Agent AI system.")
    parser.add_argument("--row", type=int, default=1, help="1-based persona row number.")
    parser.add_argument(
        "--keep-memory",
        action="store_true",
        help="Keep Chroma session data after the run for inspection.",
    )
    return parser.parse_args()


def main() -> None:
    """Run one session from the terminal."""
    args = parse_args()
    session_state = run_session(row_number=args.row, cleanup=not args.keep_memory)
    print(format_session_report(session_state))


if __name__ == "__main__":
    main()
