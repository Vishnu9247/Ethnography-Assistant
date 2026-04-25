"""LLM-based persona simulator used for testing the interview system."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List


# Allow direct execution: python simulator/persona_agent.py --row 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from utils.excel_loader import get_persona_by_row, validate_dataset_exists
from utils.llm import ask_llm
from utils.prompts import build_persona_prompt
from utils.logger import get_logger


LOGGER = get_logger("persona")


class PersonaAgent:
    """Simple stateful persona simulator."""

    def __init__(self, persona_data: Dict[str, str]):
        self.persona_data = persona_data
        self.history: List[Dict[str, str]] = []

    @classmethod
    def from_row(cls, row_number: int) -> "PersonaAgent":
        """Build a persona from one Excel row."""
        validate_dataset_exists()
        return cls(get_persona_by_row(row_number))

    def respond(self, interviewer_message: str) -> str:
        """Return a natural persona response and store the turn in memory."""
        prompt = build_persona_prompt(
            persona_name=self.persona_data["Name"],
            problem=self.persona_data["Problem"],
            person_details=self.persona_data["Person Details"],
            conversation_history=self.history,
            interviewer_message=interviewer_message,
        )
        answer = ask_llm(prompt)
        self.history.append({"interviewer": interviewer_message, "persona": answer})
        return answer


def parse_args() -> argparse.Namespace:
    """Parse terminal arguments."""
    parser = argparse.ArgumentParser(description="Run a local persona simulator.")
    parser.add_argument("--row", type=int, default=1, help="1-based row number from personas.xlsx")
    return parser.parse_args()


def main() -> None:
    """Interactive terminal loop for one persona."""
    args = parse_args()
    persona = PersonaAgent.from_row(args.row)

    print(f"Loaded persona: {persona.persona_data['Name']}")
    print("Type a message to the persona. Press Enter on an empty line to stop.")

    while True:
        user_message = input("\nInterviewer: ").strip()
        if not user_message:
            break
        answer = persona.respond(user_message)
        print(f"Persona: {answer}")


if __name__ == "__main__":
    main()
