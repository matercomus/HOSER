#!/usr/bin/env python3
"""Plot LM-TAD evaluation results (histograms + boxplot).

Reads `evaluation_results.json` (aggregated) or `evaluation_results.jsonl` and
produces per-split histograms with the computed 95th percentile threshold and
an across-splits boxplot of log-perplexities.

Usage example:
  python tools/plot_lmtad_results.py --eval-dir tools_eval_lmtad/Beijing_abnormal --out plots/lmtad_eval.png
"""

from pathlib import Path
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(eval_dir: Path):
    agg = {}
    json_file = eval_dir / "evaluation_results.json"
    jsonl_file = eval_dir / "evaluation_results.jsonl"
    if json_file.exists():
        with open(json_file, "r") as f:
            agg = json.load(f)
        return agg

    if jsonl_file.exists():
        for line in open(jsonl_file, "r"):
            try:
                rec = json.loads(line)
                split = rec.get("split")
                if split:
                    agg[split] = rec.get("results", {})
            except Exception:
                continue
        return agg

    raise FileNotFoundError(f"No results file found in {eval_dir}")


def plot_results(agg: dict, out_path: Path, show: bool = False):
    # Prepare data
    splits = sorted(agg.keys())
    data = {}
    for s in splits:
        vals = np.array(agg[s].get("log_perplexity_values", []), dtype=np.float64)
        # Filter non-finite values for plotting/statistics
        vals = vals[np.isfinite(vals)]
        data[s] = vals

    if len(splits) == 0:
        raise ValueError("No splits found in results to plot")

    # Create figure with per-split histograms and a boxplot
    sns.set(style="whitegrid")
    ncols = 2
    nrows = int(np.ceil((len(splits) + 1) / ncols))
    fig = plt.figure(figsize=(6 * ncols, 4 * nrows))

    # Histograms
    for idx, s in enumerate(splits, start=1):
        ax = fig.add_subplot(nrows, ncols, idx)
        vals = data[s]
        if vals.size == 0:
            ax.text(0.5, 0.5, "No finite values", ha="center")
            ax.set_title(s)
            continue

        sns.histplot(vals, bins=50, kde=False, ax=ax)
        thr = np.percentile(vals, 95)
        ax.axvline(thr, color="red", linestyle="--", label=f"95th pct={thr:.3f}")
        ax.set_title(f"{s} (N={len(vals)})")
        ax.set_xlabel("Log perplexity")
        ax.legend()

    # Boxplot across splits
    ax_box = fig.add_subplot(nrows, ncols, nrows * ncols)
    ordered_vals = [data[s] for s in splits]
    sns.boxplot(data=ordered_vals, orient="v", ax=ax_box)
    ax_box.set_xticklabels(splits, rotation=45, ha="right")
    ax_box.set_ylabel("Log perplexity")
    ax_box.set_title("Per-split log perplexity distribution (boxplot)")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to: {out_path}")
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot LM-TAD evaluation results")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        required=True,
        help="Directory with evaluation_results.json or .jsonl",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output image path (png/pdf)"
    )
    parser.add_argument("--show", action="store_true", help="Show plot after saving")
    args = parser.parse_args()

    agg = load_results(args.eval_dir)
    plot_results(agg, args.out, show=args.show)


if __name__ == "__main__":
    main()
