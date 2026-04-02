"""
run_all.py
──────────
Run all 5 memory experiments sequentially and print a side-by-side
comparison table of metrics.

Usage:
    python run_all.py
    python run_all.py --csv path/to/your_data.csv
"""

import sys
import argparse

from m1_no_memory        import run_experiment as run_m1
from m2_flat_vector_store import run_experiment as run_m2
from m3_graph_store      import run_experiment as run_m3
from m4_hybrid_memory    import run_experiment as run_m4
from m5_episodic_memory  import run_experiment as run_m5


def print_comparison(results: list[dict]):
    if not results:
        return

    print("\n\n" + "═" * 70)
    print("  EXPERIMENT COMPARISON — AXIS 2: MEMORY & CONTEXT STORE")
    print("═" * 70)

    header = f"{'Experiment':<30} {'Turns':>6} {'Avg Latency':>12} {'Avg Depth':>11} {'Total Time':>11}"
    print(header)
    print("─" * 70)

    for r in results:
        print(
            f"{r.get('experiment',''):<30} "
            f"{r.get('turns', 0):>6} "
            f"{str(r.get('avg_latency_s',''))+' s':>12} "
            f"{str(r.get('avg_depth_words',''))+' w':>11} "
            f"{str(r.get('total_time_s',''))+' s':>11}"
        )

    print("═" * 70)
    print("\nKey:")
    print("  Avg Latency  → lower is faster")
    print("  Avg Depth    → higher = richer, longer responses (words)")
    print("  Total Time   → total wall-clock time for all turns")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="sample_data.csv",
                        help="Path to sample data CSV")
    args = parser.parse_args()

    csv_path = args.csv
    results  = []

    experiments = [
        ("M1 — No Memory",         run_m1),
        ("M2 — Flat Vector Store",  run_m2),
        ("M3 — Graph Store",        run_m3),
        ("M4 — Hybrid Memory",      run_m4),
        ("M5 — Episodic Buffer",    run_m5),
    ]

    for name, fn in experiments:
        print(f"\n{'▶'*3}  Starting {name} ...")
        try:
            result = fn(csv_path)
            results.append(result)
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")
            results.append({"experiment": name, "error": str(e)})

    print_comparison(results)


if __name__ == "__main__":
    main()
