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

    sns.set(style="whitegrid")

    # Determine output directory and base name. If out_path is a directory
    # or has no suffix, treat it as a directory. Otherwise use parent and stem
    # to build filenames.
    if out_path.exists() and out_path.is_dir():
        out_dir = out_path
        base = "lmtad_eval"
    else:
        if out_path.suffix:
            out_dir = out_path.parent
            base = out_path.stem
        else:
            out_dir = out_path
            base = "lmtad_eval"

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save one histogram per split
    saved_files = []
    for s in splits:
        vals = data[s]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        if vals.size == 0:
            ax.text(0.5, 0.5, "No finite values", ha="center")
            ax.set_title(s)
        else:
            sns.histplot(vals, bins=50, kde=False, ax=ax)
            thr = np.percentile(vals, 95)
            ax.axvline(thr, color="red", linestyle="--", label=f"95th pct={thr:.3f}")
            ax.set_title(f"{s} (N={len(vals)})")
            ax.set_xlabel("Log perplexity")
            ax.legend()

        out_file = out_dir / f"{base}_{s}.png"
        fig.tight_layout()
        fig.savefig(out_file, dpi=150)
        saved_files.append(out_file)
        plt.close(fig)

    # Save a separate boxplot comparing splits
    fig2, ax2 = plt.subplots(figsize=(max(6, len(splits) * 0.8), 5))
    ordered_vals = [data[s] for s in splits]
    sns.boxplot(data=ordered_vals, orient="v", ax=ax2)
    ax2.set_xticklabels(splits, rotation=45, ha="right")
    ax2.set_ylabel("Log perplexity")
    ax2.set_title("Per-split log perplexity distribution (boxplot)")
    boxplot_file = out_dir / f"{base}_boxplot.png"
    fig2.tight_layout()
    fig2.savefig(boxplot_file, dpi=150)
    saved_files.append(boxplot_file)
    plt.close(fig2)

    for p in saved_files:
        print(f"Saved plot to: {p}")
    if show:
        # If user asked to show, open the boxplot
        img = plt.imread(str(boxplot_file))
        plt.imshow(img)
        plt.axis("off")
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
