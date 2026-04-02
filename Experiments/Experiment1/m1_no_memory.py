"""
m1_no_memory.py
───────────────
EXPERIMENT M1 — No Memory (Baseline)

Every turn is sent to the LLM with NO history whatsoever.
The model has zero context of what was said before.

What this measures:
  - Baseline response quality with no context
  - Drift: how quickly responses become incoherent across a session
  - Reference point to compare all other experiments against

Run:
    python m1_no_memory.py
"""

import time
from shared import (
    load_session_data,
    group_by_session,
    ask_ollama,
    print_header,
    print_turn,
    Metrics,
)


SYSTEM_PROMPT = """You are an empathetic interview assistant conducting a qualitative 
interview. Ask thoughtful follow-up questions based on the user's response."""


# ─────────────────────────────────────────────────────────
# MEMORY CLASS
# ─────────────────────────────────────────────────────────
class NoMemory:
    """
    Stateless — forgets everything after each turn.
    Each query goes to the LLM completely cold.
    """

    def get_response(self, user_query: str) -> str:
        # Only the current message, no history at all
        messages = [{"role": "user", "content": user_query}]
        return ask_ollama(messages, system_prompt=SYSTEM_PROMPT)


# ─────────────────────────────────────────────────────────
# VALIDATE WITH SAMPLE DATA
# ─────────────────────────────────────────────────────────
def run_experiment(csv_path: str = "sample_data.csv"):
    print_header("M1 — No Memory (Baseline)")

    rows     = load_session_data(csv_path)
    sessions = group_by_session(rows)
    memory   = NoMemory()
    metrics  = Metrics("M1 - No Memory")

    for session_name, turns in sessions.items():
        print(f"\n📂 Session: {session_name}  ({len(turns)} turns)")

        for i, turn in enumerate(turns, 1):
            user_query = turn["user_query"]

            start    = time.time()
            response = memory.get_response(user_query)
            latency  = time.time() - start

            metrics.record(response, latency)

            # Compare generated response vs ground truth from CSV
            ground_truth = turn["system_response"]
            print_turn(
                turn_num     = i,
                user         = user_query,
                response     = response,
                context_info = f"No context | latency={latency:.2f}s",
            )
            print(f"  GROUND TRUTH: {ground_truth[:200]}...")

    metrics.print_summary()
    return metrics.summary()


if __name__ == "__main__":
    run_experiment()
