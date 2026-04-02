"""
shared.py
─────────
Shared utilities used by all 5 memory experiments:
  - Load the sample CSV data
  - Call Ollama to get a response
  - Pretty-print conversation turns
  - Measure and compare metrics across experiments

CSV format expected:
    user_query, user_query_timestamp, system_response, system_response_timestamp, session_name
"""

import os
import csv
import time
from datetime import datetime
from typing import Optional

import ollama  # pip install ollama


# ── Config ────────────────────────────────────────────────
OLLAMA_MODEL = "wizardlm2"   # change to whichever model you have pulled
CSV_PATH     = "sample_data.csv"


# ─────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────
def load_session_data(csv_path: str = CSV_PATH) -> list[dict]:
    """
    Load the sample CSV into a list of turn dicts.

    Returns list of:
    {
        "user_query":                  str,
        "user_query_timestamp":        str,
        "system_response":             str,
        "system_response_timestamp":   str,
        "session_name":                str,
    }
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Sample data not found at '{csv_path}'.\n"
            "Create a CSV with columns: user_query, user_query_timestamp, "
            "system_response, system_response_timestamp, session_name"
        )

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v.strip() for k, v in row.items()})

    print(f"✅ Loaded {len(rows)} turns from '{csv_path}'")
    return rows


def group_by_session(rows: list[dict]) -> dict[str, list[dict]]:
    """Group turns by session_name."""
    sessions: dict[str, list[dict]] = {}
    for row in rows:
        name = row["session_name"]
        sessions.setdefault(name, []).append(row)
    return sessions


# ─────────────────────────────────────────────────────────
# OLLAMA CALLER
# ─────────────────────────────────────────────────────────
def ask_ollama(messages: list[dict], system_prompt: str = "") -> str:
    """
    Send a list of {role, content} messages to Ollama and return the reply.

    Parameters
    ----------
    messages      : conversation history in OpenAI format
    system_prompt : optional system instruction prepended to the conversation
    """
    full_messages = []
    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})
    full_messages.extend(messages)

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=full_messages,
        )
        return response["message"]["content"].strip()
    except Exception as e:
        return f"[Ollama error: {e}]"


# ─────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────
class Metrics:
    """Track response quality metrics for one experiment run."""

    def __init__(self, experiment_name: str):
        self.name       = experiment_name
        self.latencies  : list[float] = []
        self.depths     : list[int]   = []   # response word count
        self.turns      : int = 0

    def record(self, response: str, latency: float):
        self.turns     += 1
        self.latencies.append(latency)
        self.depths.append(len(response.split()))

    def summary(self) -> dict:
        if not self.latencies:
            return {}
        return {
            "experiment"    : self.name,
            "turns"         : self.turns,
            "avg_latency_s" : round(sum(self.latencies) / len(self.latencies), 2),
            "avg_depth_words": round(sum(self.depths) / len(self.depths), 1),
            "total_time_s"  : round(sum(self.latencies), 2),
        }

    def print_summary(self):
        s = self.summary()
        print(f"\n{'═'*50}")
        print(f"  METRICS — {s['experiment']}")
        print(f"{'═'*50}")
        print(f"  Turns processed   : {s['turns']}")
        print(f"  Avg latency/turn  : {s['avg_latency_s']}s")
        print(f"  Avg response depth: {s['avg_depth_words']} words")
        print(f"  Total time        : {s['total_time_s']}s")
        print(f"{'═'*50}\n")


# ─────────────────────────────────────────────────────────
# PRETTY PRINTER
# ─────────────────────────────────────────────────────────
def print_turn(turn_num: int, user: str, response: str, context_info: str = ""):
    print(f"\n{'─'*50}")
    print(f"  Turn {turn_num}")
    if context_info:
        print(f"  [{context_info}]")
    print(f"  USER     : {user[:120]}{'...' if len(user)>120 else ''}")
    print(f"  RESPONSE : {response[:300]}{'...' if len(response)>300 else ''}")


def print_header(title: str):
    print(f"\n{'═'*60}")
    print(f"  EXPERIMENT: {title}")
    print(f"{'═'*60}")
