"""Run the full LangGraph pipeline on one row or all rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config import RESULTS_DIR, ensure_directories
from evaluation.metrics import score_agent1, score_agent2, score_agent3, score_full_system
from evaluation.plots import create_plots
from main import format_session_report, run_session
from utils.excel_loader import get_all_personas


def evaluate_row(row_number: int) -> Dict[str, float]:
    """Run the full system and compute combined metrics."""
    state = run_session(row_number=row_number, cleanup=True)

    agent1_scores = score_agent1(state["intake_result"], state["persona_data"])
    agent2_scores = score_agent2(state["ethnography_result"], state["persona_data"])
    agent3_scores = score_agent3(state["analysis_result"], state["persona_data"])
    full_scores = score_full_system(agent1_scores, agent2_scores, agent3_scores)

    final_scores = {
        "row": row_number,
        "name": state["persona_data"]["Name"],
        **agent1_scores,
        **agent2_scores,
        **agent3_scores,
        **full_scores,
    }
    state["scores"] = final_scores

    print(format_session_report(state))
    return final_scores


def save_results(rows: List[Dict[str, float]]) -> Path:
    """Save the full-system metrics as CSV."""
    ensure_directories()
    output_path = RESULTS_DIR / "full_system_scores.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse terminal arguments."""
    parser = argparse.ArgumentParser(description="Evaluate the full system.")
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
    create_plots()
    print(f"Saved full system results to {output_path}")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
