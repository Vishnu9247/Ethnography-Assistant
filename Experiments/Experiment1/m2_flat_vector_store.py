"""
m2_flat_vector_store.py
───────────────────────
EXPERIMENT M2 — Flat Vector Store (ChromaDB)

Every turn (user query + system response) is embedded and stored in ChromaDB.
Before generating a reply, the top-K most semantically similar past turns
are retrieved and injected into the LLM's context window.

What this measures:
  - Whether semantic retrieval improves follow-up relevance
  - How well the model references past emotional moments
  - Recall accuracy: does the right past turn get retrieved?

Install:
    pip install chromadb sentence-transformers

Run:
    python m2_flat_vector_store.py
"""

import time
import uuid
import chromadb
from chromadb.utils import embedding_functions

from shared import (
    load_session_data,
    group_by_session,
    ask_ollama,
    print_header,
    print_turn,
    Metrics,
)


SYSTEM_PROMPT = """You are an empathetic interview assistant conducting a qualitative 
interview. Use the retrieved past context to ask relevant, personalised follow-up 
questions. Reference what the participant said earlier when appropriate."""

TOP_K = 3   # number of similar past turns to retrieve


# ─────────────────────────────────────────────────────────
# MEMORY CLASS
# ─────────────────────────────────────────────────────────
class FlatVectorMemory:
    """
    Stores every conversation turn as an embedding in ChromaDB.
    On each new query, retrieves the TOP_K most relevant past turns.
    """

    def __init__(self, session_name: str):
        self.session_name = session_name

        # In-memory ChromaDB (no disk writes — resets each run)
        # To persist: chromadb.PersistentClient(path="./chroma_db")
        self.client = chromadb.Client()

        # Use a local sentence-transformer for embeddings (no API key needed)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Each session gets its own isolated collection
        safe_name = session_name.replace(" ", "_").lower()
        self.collection = self.client.create_collection(
            name=f"session_{safe_name}_{uuid.uuid4().hex[:6]}",
            embedding_function=self.embed_fn,
        )

        self.turn_count = 0

    def store_turn(self, user_query: str, system_response: str):
        """Embed and store a completed turn."""
        self.turn_count += 1
        combined = f"User: {user_query}\nAssistant: {system_response}"
        self.collection.add(
            documents=[combined],
            ids=[f"turn_{self.turn_count}"],
            metadatas=[{"turn": self.turn_count, "session": self.session_name}],
        )

    def retrieve_context(self, user_query: str) -> str:
        """Retrieve TOP_K most semantically similar past turns."""
        if self.turn_count == 0:
            return ""

        k = min(TOP_K, self.turn_count)
        results = self.collection.query(
            query_texts=[user_query],
            n_results=k,
        )

        retrieved = results["documents"][0]  # list of matching docs
        context   = "\n\n".join(
            f"[Past turn {i+1}]: {doc}" for i, doc in enumerate(retrieved)
        )
        return context

    def get_response(self, user_query: str) -> tuple[str, str]:
        """
        Retrieve relevant context, build prompt, get LLM response.

        Returns
        -------
        (response, retrieved_context)
        """
        context = self.retrieve_context(user_query)

        if context:
            augmented_query = (
                f"Relevant past context from this session:\n{context}\n\n"
                f"Current user message: {user_query}"
            )
        else:
            augmented_query = user_query

        messages  = [{"role": "user", "content": augmented_query}]
        response  = ask_ollama(messages, system_prompt=SYSTEM_PROMPT)

        # Store the turn AFTER generating response
        self.store_turn(user_query, response)

        return response, context


# ─────────────────────────────────────────────────────────
# VALIDATE WITH SAMPLE DATA
# ─────────────────────────────────────────────────────────
def run_experiment(csv_path: str = "sample_data.csv"):
    print_header("M2 — Flat Vector Store (ChromaDB)")

    rows     = load_session_data(csv_path)
    sessions = group_by_session(rows)
    metrics  = Metrics("M2 - Flat Vector Store")

    for session_name, turns in sessions.items():
        print(f"\n📂 Session: {session_name}  ({len(turns)} turns)")
        memory = FlatVectorMemory(session_name)

        # ── Pre-load: store historical turns from CSV into vector store ──
        # This simulates a session that already has prior context.
        # In a live interview, turns are stored in real time.
        for turn in turns[:-1]:  # store all but the last turn
            memory.store_turn(turn["user_query"], turn["system_response"])
        print(f"   Pre-loaded {memory.turn_count} historical turns into ChromaDB.")

        # ── Test on remaining turns ──────────────────────────────────────
        for i, turn in enumerate(turns, 1):
            user_query = turn["user_query"]

            start               = time.time()
            response, context   = memory.get_response(user_query)
            latency             = time.time() - start

            metrics.record(response, latency)

            retrieved_count = context.count("[Past turn") if context else 0
            ground_truth    = turn["system_response"]

            print_turn(
                turn_num     = i,
                user         = user_query,
                response     = response,
                context_info = f"Retrieved {retrieved_count} past turns | latency={latency:.2f}s",
            )
            print(f"  GROUND TRUTH : {ground_truth[:200]}...")
            if context:
                print(f"  RETRIEVED CTX: {context[:300]}...")

    metrics.print_summary()
    return metrics.summary()


if __name__ == "__main__":
    run_experiment()
