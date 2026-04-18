"""
pipeline.py — Master Orchestrator
Ties all agents together: Optimus Prime → Bumblebee → Megatron.

Usage:
  python pipeline.py                          # Full pipeline
  python pipeline.py --agent optimus          # Run only Optimus Prime
  python pipeline.py --agent bumblebee <doc>  # Run only Bumblebee (requires intake doc path)
  python pipeline.py --agent megatron <doc>   # Run only Megatron (requires intake doc path)
"""

import sys
import argparse
import traceback


def run_full_pipeline():
    print("\n" + "=" * 70)
    print("  THE TRANSFORMERS PIPELINE — ETHNOGRAPHIC RESEARCH SYSTEM")
    print("=" * 70)

    # ── Stage 1: Optimus Prime ────────────────────────────────────────────────
    print("\n[STAGE 1] Optimus Prime — Problem Definition")
    print("-" * 50)
    try:
        import optimus_prime
        intake_path = optimus_prime.run()
        if not intake_path:
            print("[Pipeline] ERROR: Optimus Prime did not produce an intake document. Aborting.")
            sys.exit(1)
        print(f"[Pipeline] ✓ Intake document: {intake_path}")
    except Exception as e:
        print(f"[Pipeline] FATAL ERROR in Optimus Prime:\n{traceback.format_exc()}")
        sys.exit(1)

    # ── Stage 2: Bumblebee ────────────────────────────────────────────────────
    print("\n[STAGE 2] Bumblebee — Field Exploration")
    print("-" * 50)
    try:
        import bumblebee
        bumblebee.run(intake_path)
        print("[Pipeline] ✓ All fields explored and stored in Cybertron.")
    except FileNotFoundError as e:
        print(f"[Pipeline] ERROR: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"[Pipeline] ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[Pipeline] FATAL ERROR in Bumblebee:\n{traceback.format_exc()}")
        sys.exit(1)

    # ── Stage 3: Megatron ─────────────────────────────────────────────────────
    print("\n[STAGE 3] Megatron — Synthesis & Solution")
    print("-" * 50)
    try:
        import megatron
        report_path = megatron.run(intake_path)
        if not report_path:
            print("[Pipeline] WARNING: Megatron did not produce a report.")
        else:
            print(f"[Pipeline] ✓ Report: {report_path}")
    except Exception as e:
        print(f"[Pipeline] FATAL ERROR in Megatron:\n{traceback.format_exc()}")
        sys.exit(1)

    # ── Final Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Intake Document : {intake_path}")
    print(f"  Solution Report : {report_path}")
    print("=" * 70 + "\n")


def run_single_agent(agent: str, doc_path: str = None):
    if agent == "optimus":
        import optimus_prime
        path = optimus_prime.run()
        print(f"\n[Pipeline] Optimus Prime complete. Output: {path}")

    elif agent == "bumblebee":
        if not doc_path:
            print("ERROR: --agent bumblebee requires a path to the intake document.")
            print("Usage: python pipeline.py --agent bumblebee <path>")
            sys.exit(1)
        import bumblebee
        bumblebee.run(doc_path)
        print("\n[Pipeline] Bumblebee complete.")

    elif agent == "megatron":
        if not doc_path:
            print("ERROR: --agent megatron requires a path to the intake document.")
            print("Usage: python pipeline.py --agent megatron <path>")
            sys.exit(1)
        import megatron
        path = megatron.run(doc_path)
        print(f"\n[Pipeline] Megatron complete. Report: {path}")

    else:
        print(f"ERROR: Unknown agent '{agent}'. Choose from: optimus, bumblebee, megatron")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="The Transformers Pipeline — Ethnographic Research System"
    )
    parser.add_argument(
        "--agent",
        choices=["optimus", "bumblebee", "megatron"],
        help="Run a single agent in isolation",
        default=None,
    )
    parser.add_argument(
        "doc_path",
        nargs="?",
        help="Path to intake document (required for bumblebee and megatron)",
        default=None,
    )
    args = parser.parse_args()

    if args.agent:
        run_single_agent(args.agent, args.doc_path)
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
