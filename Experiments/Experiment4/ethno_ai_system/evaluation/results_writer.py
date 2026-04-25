"""Helpers for saving session conversations and reports."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

from config import RESULTS_DIR, ensure_directories
from main import format_session_report


def _safe_name(text: str) -> str:
    """Convert a name into a filename-friendly version."""
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return cleaned.strip("_") or "unknown_persona"


def build_session_artifact(session_state: Dict[str, Any]) -> Dict[str, Any]:
    """Create one JSON-friendly artifact with all session details."""
    return {
        "session_id": session_state["session_id"],
        "row": session_state["persona_row"],
        "persona_name": session_state["persona_data"]["Name"],
        "persona_data": session_state["persona_data"],
        "agent1": {
            "conversation": session_state["intake_result"].get("conversation", []),
            "problem_summary": session_state["intake_result"].get("problem_summary", ""),
            "final_problem_statement": session_state["intake_result"].get(
                "final_problem_statement", ""
            ),
            "key_signals": session_state["intake_result"].get("key_signals", []),
        },
        "agent2": {
            "selected_domains": session_state["ethnography_result"].get("selected_domains", []),
            "domain_conversations": session_state["ethnography_result"].get(
                "domain_conversations", {}
            ),
            "domain_summaries": session_state["ethnography_result"].get("domain_summaries", {}),
            "redundant_questions": session_state["ethnography_result"].get(
                "redundant_questions", 0
            ),
        },
        "agent3": {
            "retrieved_items": session_state["analysis_result"].get("retrieved_items", []),
            "patterns": session_state["analysis_result"].get("patterns", []),
            "root_causes": session_state["analysis_result"].get("root_causes", []),
            "recommendations": session_state["analysis_result"].get("recommendations", []),
        },
        "persona_history": session_state.get("persona_history", []),
        "scores": session_state.get("scores", {}),
    }


def save_session_artifacts(session_state: Dict[str, Any]) -> Dict[str, Path]:
    """Save transcript-style outputs for one row inside results/."""
    ensure_directories()

    conversations_dir = RESULTS_DIR / "conversations"
    reports_dir = RESULTS_DIR / "reports"
    conversations_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    row_number = session_state["persona_row"]
    persona_name = session_state["persona_data"]["Name"]
    file_stem = f"row_{row_number}_{_safe_name(persona_name)}"

    artifact = build_session_artifact(session_state)
    json_path = conversations_dir / f"{file_stem}.json"
    txt_path = reports_dir / f"{file_stem}_report.txt"

    json_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    txt_path.write_text(format_session_report(session_state), encoding="utf-8")

    jsonl_path = RESULTS_DIR / "full_system_conversations.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(artifact) + "\n")

    return {
        "json_path": json_path,
        "report_path": txt_path,
        "jsonl_path": jsonl_path,
    }
