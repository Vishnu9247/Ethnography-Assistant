"""Run Agent 1 on one row or all rows."""

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
from config import RESULTS_DIR, ensure_directories
from evaluation.metrics import score_agent1
from simulator.persona_agent import PersonaAgent
from utils.excel_loader import get_all_personas, get_persona_by_row


def evaluate_row(row_number: int) -> Dict[str, float]:
    """Run Agent 1 and compute metrics for one row."""
    persona_data = get_persona_by_row(row_number)
    persona_agent = PersonaAgent(persona_data)
    result = run_agent1(persona_agent, persona_data, session_id=f"agent1_row_{row_number}")
    scores = score_agent1(result, persona_data)
    scores["row"] = row_number
    scores["name"] = persona_data["Name"]
    return scores


def save_results(rows: List[Dict[str, float]]) -> Path:
    """Save Agent 1 metrics as CSV."""
    ensure_directories()
    output_path = RESULTS_DIR / "agent1_scores.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse terminal arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Agent 1.")
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
    print(f"Saved Agent 1 results to {output_path}")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
