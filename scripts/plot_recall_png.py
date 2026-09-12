"""Renders dflash_topk_recall.csv (from analyze_dflash_topk_recall.py) as a
static PNG chart for embedding in README.md (the HTML report already has an
interactive inline-SVG version; this is just a portable image).

Usage:
    uv run --with matplotlib python scripts/plot_recall_png.py --run-name llama3.1-8b
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
DOCS_IMG_DIR = Path(__file__).resolve().parents[1] / "docs" / "img"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    rows = []
    with open(RESULTS_DIR / args.run_name / "dflash_topk_recall.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    ks = [int(r["k"]) for r in rows]
    post = [float(r["recall_post_reject"]) for r in rows]
    pre = [float(r["recall_pre_reject"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ks, post, marker="o", markersize=5, linewidth=2, color="#2a78d6",
            label="post-reject (reject slot + rest of block)")
    ax.plot(ks, pre, marker="o", markersize=5, linewidth=2, color="#898781",
            label="pre-reject (sanity check)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k")
    ax.set_ylabel("recall@k")
    ax.set_ylim(0, 1.05)
    ax.set_title("DFlash: is the verifier's token in the drafter's top-k?", fontsize=12, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()

    DOCS_IMG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DOCS_IMG_DIR / "recall_at_k.png"
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
