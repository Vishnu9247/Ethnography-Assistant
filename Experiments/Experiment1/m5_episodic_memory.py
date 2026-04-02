"""
m5_episodic_memory.py
─────────────────────
EXPERIMENT M5 — Episodic Memory Buffer

Conversation turns are grouped into "episodes" of N turns each.
After every episode, the LLM itself summarises what was discussed into a
compact memory summary. Future turns receive both the recent raw turns AND
all past episode summaries.

This solves the context window problem: instead of stuffing 50 raw turns
into the prompt, you pass compressed summaries + only the recent raw turns.

What this measures:
  - Does compression lose important detail vs raw retrieval?
  - Is the episode summary accurate enough to guide future questions?
  - Performance at long sessions (20+ turns) where other methods break down

Run:
    python m5_episodic_memory.py
"""

import time
from dataclasses import dataclass, field
from shared import (
    load_session_data,
    group_by_session,
    ask_ollama,
    print_header,
    print_turn,
    Metrics,
)


SYSTEM_PROMPT = """You are an empathetic interview assistant conducting a qualitative 
interview. You have a summary of what was discussed in previous episodes, plus the 
recent conversation. Use both to ask deeply contextual follow-up questions."""

EPISODE_SIZE = 3  # compress every N turns into a summary


# ─────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────
@dataclass
class Turn:
    role   : str   # "user" or "assistant"
    content: str


@dataclass
class Episode:
    number : int
    turns  : list[Turn]
    summary: str = ""   # filled in after compression


# ─────────────────────────────────────────────────────────
# MEMORY CLASS
# ─────────────────────────────────────────────────────────
class EpisodicMemory:
    """
    Maintains a rolling buffer of recent turns + compressed episode summaries.

    Structure:
        past_episodes  : list of Episode (each with a summary)
        current_buffer : list of Turn (raw, not yet summarised)

    When current_buffer reaches EPISODE_SIZE turns, it is compressed into
    a new Episode summary and cleared.
    """

    def __init__(self, session_name: str, episode_size: int = EPISODE_SIZE):
        self.session_name   = session_name
        self.episode_size   = episode_size
        self.past_episodes  : list[Episode] = []
        self.current_buffer : list[Turn]    = []
        self.total_turns    = 0

    def _compress_buffer(self):
        """Summarise the current buffer into an episode and clear it."""
        if not self.current_buffer:
            return

        # Build a transcript for the LLM to summarise
        transcript = "\n".join(
            f"{t.role.upper()}: {t.content}" for t in self.current_buffer
        )

        summary_prompt = (
            f"Summarise this interview excerpt in 2-3 sentences. "
            f"Focus on key topics, emotions, and any important personal details "
            f"that should inform future questions.\n\n{transcript}"
        )

        summary = ask_ollama(
            messages=[{"role": "user", "content": summary_prompt}],
            system_prompt="You are a concise interview analyst.",
        )

        episode = Episode(
            number  = len(self.past_episodes) + 1,
            turns   = list(self.current_buffer),
            summary = summary,
        )
        self.past_episodes.append(episode)
        self.current_buffer.clear()

        print(f"   📦 Episode {episode.number} compressed → "
              f"'{summary[:100]}...'")

    def store_turn(self, user_query: str, system_response: str):
        """Add a completed turn to the buffer; compress if full."""
        self.total_turns   += 1
        self.current_buffer.append(Turn("user",      user_query))
        self.current_buffer.append(Turn("assistant", system_response))

        if len(self.current_buffer) >= self.episode_size * 2:
            self._compress_buffer()

    def build_context(self) -> str:
        """
        Build the context string passed to the LLM:
          1. Past episode summaries (compressed)
          2. Current buffer (raw recent turns)
        """
        parts = []

        if self.past_episodes:
            summaries = "\n".join(
                f"Episode {ep.number}: {ep.summary}"
                for ep in self.past_episodes
            )
            parts.append(f"=== Past Episode Summaries ===\n{summaries}")

        if self.current_buffer:
            recent = "\n".join(
                f"{t.role.upper()}: {t.content}"
                for t in self.current_buffer
            )
            parts.append(f"=== Recent Conversation ===\n{recent}")

        return "\n\n".join(parts)

    def get_response(self, user_query: str) -> tuple[str, str]:
        """Build episodic context, get LLM response."""
        context = self.build_context()

        if context:
            augmented_query = (
                f"{context}\n\n"
                f"Current user message: {user_query}"
            )
        else:
            augmented_query = user_query

        messages = [{"role": "user", "content": augmented_query}]
        response = ask_ollama(messages, system_prompt=SYSTEM_PROMPT)

        self.store_turn(user_query, response)
        return response, context

    def memory_stats(self) -> str:
        return (
            f"Episodes: {len(self.past_episodes)} | "
            f"Buffer: {len(self.current_buffer)//2} raw turns"
        )


# ─────────────────────────────────────────────────────────
# VALIDATE WITH SAMPLE DATA
# ─────────────────────────────────────────────────────────
def run_experiment(csv_path: str = "sample_data.csv"):
    print_header("M5 — Episodic Memory Buffer")

    rows     = load_session_data(csv_path)
    sessions = group_by_session(rows)
    metrics  = Metrics("M5 - Episodic Memory")

    for session_name, turns in sessions.items():
        print(f"\n📂 Session: {session_name}  ({len(turns)} turns)")
        print(f"   Episode size: every {EPISODE_SIZE} turns → compressed to summary")

        memory = EpisodicMemory(session_name, episode_size=EPISODE_SIZE)

        # Pre-load historical turns from CSV
        # These will automatically be compressed into episode summaries
        print("\n   Pre-loading history (watch for episode compressions)...")
        for turn in turns[:-1]:
            memory.store_turn(turn["user_query"], turn["system_response"])
        print(f"   Pre-load done.  {memory.memory_stats()}")

        # Test on each turn
        for i, turn in enumerate(turns, 1):
            user_query = turn["user_query"]

            start             = time.time()
            response, context = memory.get_response(user_query)
            latency           = time.time() - start

            metrics.record(response, latency)

            ground_truth  = turn["system_response"]
            ep_count      = len(memory.past_episodes)
            buf_size      = len(memory.current_buffer) // 2

            print_turn(
                turn_num     = i,
                user         = user_query,
                response     = response,
                context_info = (
                    f"{ep_count} episode summaries + {buf_size} raw turns | "
                    f"latency={latency:.2f}s"
                ),
            )
            print(f"  GROUND TRUTH : {ground_truth[:200]}...")
            if memory.past_episodes:
                last_summary = memory.past_episodes[-1].summary
                print(f"  LAST EPISODE : {last_summary[:200]}...")

    metrics.print_summary()
    return metrics.summary()


if __name__ == "__main__":
    run_experiment()
