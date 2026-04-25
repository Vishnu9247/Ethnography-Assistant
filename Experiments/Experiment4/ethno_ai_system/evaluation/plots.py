"""Create simple matplotlib charts from result CSV files."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from config import RESULTS_DIR, ensure_directories


def create_plots() -> None:
    """Create summary charts if the full system CSV exists."""
    ensure_directories()
    source_file = RESULTS_DIR / "full_system_scores.csv"
    if not source_file.exists():
        return

    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source_file)
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return

    averages = numeric_df.mean(numeric_only=True).sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    averages.plot(kind="bar", color="steelblue")
    plt.title("Average KPI Scores")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(plots_dir / "summary_metrics.png")
    plt.close()

    if "row" in df.columns and "full_system_average_score" in df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(df["row"], df["full_system_average_score"], marker="o", color="darkgreen")
        plt.title("Row-wise Full System Performance")
        plt.xlabel("Row")
        plt.ylabel("Average Score")
        plt.tight_layout()
        plt.savefig(plots_dir / "rowwise_performance.png")
        plt.close()


if __name__ == "__main__":
    create_plots()
    print("Plot generation complete.")
