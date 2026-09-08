"""Export publication-sized figures from the Method 2 offline report tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def plot(tables_path: Path, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    report = json.loads(tables_path.read_text(encoding="utf-8"))
    rows = report["rows"]
    output.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for offset, key, label, color in [(-0.2, "normalized_arga", "Pipeline", "#2563eb"),
                                     (0.2, "oracle_arga", "Oracle tool", "#14b8a6")]:
        ax.bar(x + offset, [r[key] * 100 for r in rows], width=0.36, label=label, color=color)
    estimates = [r["normalized_arga"] * 100 for r in rows]
    errors = [[(r["normalized_arga"] - r["ci"]["lower"]) * 100 for r in rows],
              [(r["ci"]["upper"] - r["normalized_arga"]) * 100 for r in rows]]
    ax.errorbar(x - 0.2, estimates, yerr=errors, fmt="none", color="#172554", capsize=4)
    ax.set_xticks(x, ["Benchmark", "Custom seen", "Custom unseen"])
    ax.set_ylabel("Normalized exact call accuracy on positives (%)")
    ax.set_ylim(0, 100)
    title = "legacy run with unseen negative exposure" if report["protocol"].startswith("legacy") else report["protocol"].replace("_", " ")
    ax.set_title(f"Method 2 — {title}")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"oracle_pipeline.{suffix}", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(args.tables, args.output)


if __name__ == "__main__":
    main()
