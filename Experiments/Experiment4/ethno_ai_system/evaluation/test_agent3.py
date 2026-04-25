"""Run Agent 3 on one row or all rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.agent1_intake import run_agent1
from agents.agent2_ethnography import run_agent2
from agents.agent3_analysis import run_agent3
from config import RESULTS_DIR, ensure_directories
from evaluation.metrics import score_agent3
from memory.vector_store import delete_session, init_store
from simulator.persona_agent import PersonaAgent
from utils.excel_loader import get_all_personas, get_persona_by_row


def evaluate_row(row_number: int) -> Dict[str, float]:
    """Run Agent 3 and compute metrics for one row."""
    init_store()
    persona_data = get_persona_by_row(row_number)
    persona_agent = PersonaAgent(persona_data)
    session_id = f"agent3_row_{row_number}"

    intake_result = run_agent1(persona_agent, persona_data, session_id=session_id)
    ethnography_result = run_agent2(persona_agent, persona_data, intake_result, session_id=session_id)
    result = run_agent3(persona_data, intake_result, ethnography_result, session_id=session_id)
    scores = score_agent3(result, persona_data)
    scores["row"] = row_number
    scores["name"] = persona_data["Name"]
    delete_session(session_id)
    return scores


def save_results(rows: List[Dict[str, float]]) -> Path:
    """Save Agent 3 metrics as CSV."""
    ensure_directories()
    output_path = RESULTS_DIR / "agent3_scores.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse terminal arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Agent 3.")
    parser.add_argument("--row", type=int, help="1-based row number")
    parser.add_argument("--all", action="store_true", help="Evaluate all rows")
    return parser.parse_args()


def main() -> None:
    """Run the evaluation from the terminal."""
    args = parse_args()

    if args.all:
        results = [evaluate_row(index) for index in range(1, len(get_all_personas()) + 1)]
    else:
        row_number = args.row or 1
        results = [evaluate_row(row_number)]

    output_path = save_results(results)
    print(f"Saved Agent 3 results to {output_path}")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
