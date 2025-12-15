#!/usr/bin/env python3
"""Diagnostics for LM-TAD detection performance vs injected anomalies.

Produces:
- detection_at_injected_rate.csv (metrics at top-13% and top-15%)
- pr_curve_<split>.png, pr_data_<split>.csv, pr_summary.csv
- score_distributions_<split>.png, score_distribution_stats.csv

Usage:
  python tools/scripts/diagnose_detection.py --eval-dir tools_eval_lmtad/Beijing_abnormal_2 \
      --data-dir data/Beijing_abnormal_2 --out-dir tools_eval_lmtad/Beijing_abnormal_2

This script is defensive: it will try to match injected rows by `row_index` or `trajectory_id`.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import precision_recall_curve, average_precision_score
except Exception:
    print("scikit-learn is required. Install with `uv add scikit-learn` or pip.")
    raise

try:
    from scipy import stats
except Exception:
    stats = None


def load_eval_results(eval_dir):
    p = Path(eval_dir)
    jsonl = p / "evaluation_results.jsonl"
    results = []
    if jsonl.exists():
        with jsonl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    results.append(json.loads(line))
                except Exception:
                    continue
        return results

    j = p / "evaluation_results.json"
    if j.exists():
        with j.open() as f:
            data = json.load(f)
            if isinstance(data, list):
                # possible form: list of {"split": ..., "results": [...]}
                # Flatten if necessary
                flat = []
                for item in data:
                    if isinstance(item, dict) and "split" in item and "results" in item:
                        res = item["results"]
                        if isinstance(res, list):
                            for r in res:
                                if isinstance(r, dict):
                                    r["split"] = item["split"]
                                    flat.append(r)
                        elif isinstance(res, dict) and "log_perplexity_values" in res:
                            lp = res.get("log_perplexity_values", [])
                            for i, v in enumerate(lp):
                                flat.append(
                                    {
                                        "row_index": i,
                                        "log_perplexity_values": [v],
                                        "split": item["split"],
                                    }
                                )
                        else:
                            flat.append(item)
                    else:
                        flat.append(item)
                return flat
            # try common keys
            # If this file is a dict of splits -> metrics (common form),
            # and each split contains `log_perplexity_values`, flatten to per-trajectory entries.
            if isinstance(data, dict):
                splits = []
                for split_name, split_val in data.items():
                    if (
                        isinstance(split_val, dict)
                        and "log_perplexity_values" in split_val
                    ):
                        lp = split_val.get("log_perplexity_values", [])
                        for i, v in enumerate(lp):
                            splits.append(
                                {
                                    "row_index": i,
                                    "log_perplexity_values": [v],
                                    "split": split_name,
                                }
                            )
                if splits:
                    print(
                        f"Loaded {len(splits)} flattened per-trajectory entries from {j}"
                    )
                    return splits
            for key in ("rows", "per_trajectory", "results", "trajectories"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            # otherwise try to infer items
            # if dict of lists -> join them
            items = []
            for v in data.values():
                if isinstance(v, list):
                    items.extend(v)
            if items:
                print(f"Loaded {len(items)} items from {j}")
                return items
    return results


def extract_score_and_id(entry):
    # Try to get a scalar score and an id/index and split
    score = None
    idx = None
    split = entry.get("split") or entry.get("split_name") or entry.get("phase")
    # score fields
    for k in ("log_perplexity_values", "score", "scores", "log_perplexity"):
        if k in entry:
            v = entry[k]
            if isinstance(v, list) and v:
                # reduce to mean
                try:
                    score = float(np.mean(v))
                except Exception:
                    score = float(v[0])
            else:
                try:
                    score = float(v)
                except Exception:
                    pass
            break
    # id fields
    for k in ("row_index", "index", "trajectory_index", "trajectory_id", "id"):
        if k in entry:
            idx = entry[k]
            break
    return idx, score, split


def load_injected_map(data_dir):
    # find *.injected_indices.jsonl files and parse per-split
    d = Path(data_dir)
    mapping = {}
    for p in d.glob("*.injected_indices.jsonl"):
        split = p.name.split(".")[0]
        injected = {}
        strong_map = {}
        with p.open() as f:
            for line in f:
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                # expect row_index or index or id
                row = (
                    j.get("row_index")
                    if "row_index" in j
                    else j.get("idx")
                    if "idx" in j
                    else j.get("index")
                    if "index" in j
                    else j.get("id")
                )
                if row is None:
                    # try to infer from position? skip
                    continue
                # normalize key types to str and int for flexible matching
                injected[row] = True
                injected[str(row)] = True
                try:
                    injected[int(row)] = True
                except Exception:
                    pass
                if j.get("strong"):
                    strong_map[row] = True
                    strong_map[str(row)] = True
                    try:
                        strong_map[int(row)] = True
                    except Exception:
                        pass
        mapping[split] = {"injected": injected, "strong": strong_map}
    return mapping


def match_and_build_table(eval_entries, injected_map):
    # Build dict: split -> DataFrame with columns id, score, is_injected, is_strong
    tables = {}
    for e in eval_entries:
        idx, score, split = extract_score_and_id(e)
        if score is None:
            continue
        if split is None:
            split = "train"
        if split not in tables:
            tables[split] = []
        tables[split].append({"id": idx, "score": score, "raw": e})

    out = {}
    for split, rows in tables.items():
        df = pd.DataFrame(rows)
        # create mapping: try numeric indices first
        inj = injected_map.get(split, {}).get("injected", {})
        strong = injected_map.get(split, {}).get("strong", {})
        if not inj:
            # try fallback: if only one split in injected_map, use it
            if len(injected_map) == 1:
                inj = list(injected_map.values())[0].get("injected", {})
                strong = list(injected_map.values())[0].get("strong", {})

        # try to mark by id exact match. Normalize df['id'] to str and int where possible.
        if not df["id"].isnull().all():

            def check_injected(x):
                if x in inj:
                    return True
                sx = str(x)
                if sx in inj:
                    return True
                try:
                    ix = int(x)
                    if ix in inj:
                        return True
                except Exception:
                    pass
                return False

            def check_strong(x):
                if x in strong:
                    return True
                if str(x) in strong:
                    return True
                try:
                    if int(x) in strong:
                        return True
                except Exception:
                    pass
                return False

            df["is_injected"] = df["id"].apply(lambda x: bool(check_injected(x)))
            df["is_strong"] = df["id"].apply(lambda x: bool(check_strong(x)))
        else:
            # no id available: cannot match
            df["is_injected"] = False
            df["is_strong"] = False

        out[split] = df
    return out


def compute_detection_at_injected(df, out_dir, dataset_name):
    rows = []
    os.makedirs(out_dir, exist_ok=True)
    for split, g in df.items():
        scores = g["score"].values
        is_inj = g["is_injected"].values.astype(bool)
        n_total = len(scores)
        n_inj = int(is_inj.sum())
        inj_frac = n_inj / max(1, n_total)
        for p in (13, 15):
            # top-p percent => threshold at 100-p percentile
            thr = np.percentile(scores, 100.0 - p)
            pred = scores > thr
            TP = int((pred & is_inj).sum())
            FP = int((pred & ~is_inj).sum())
            FN = int((~pred & is_inj).sum())
            TN = int((~pred & ~is_inj).sum())
            precision = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
            recall = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else float("nan")
            )
            acc = (TP + TN) / max(1, (TP + TN + FP + FN))
            pred_rate = (TP + FP) / max(1, n_total)
            rows.append(
                {
                    "dataset": dataset_name,
                    "split": split,
                    "threshold_pct": p,
                    "threshold_value": float(thr),
                    "TP": TP,
                    "FP": FP,
                    "FN": FN,
                    "TN": TN,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "accuracy": acc,
                    "n_total": n_total,
                    "n_injected": n_inj,
                    "injected_fraction": inj_frac,
                    "predicted_rate": pred_rate,
                }
            )

        # save small barplot
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(4, 3))
            vals = [r["recall"] for r in rows if r["split"] == split]
            precs = [r["precision"] for r in rows if r["split"] == split]
            labels = [f"top-{r['threshold_pct']}%" for r in rows if r["split"] == split]
            x = np.arange(len(labels))
            ax.bar(
                x - 0.15,
                [0 if math.isnan(v) else v for v in precs],
                width=0.3,
                label="precision",
            )
            ax.bar(
                x + 0.15,
                [0 if math.isnan(v) else v for v in vals],
                width=0.3,
                label="recall",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylim(0, 1)
            ax.set_title(f"Precision/Recall at top-p ({split})")
            ax.legend()
            fig.tight_layout()
            figpath = Path(out_dir) / f"detection_at_injected_rate_{split}.png"
            fig.savefig(figpath)
            plt.close(fig)
        except Exception:
            pass

    out_csv = Path(out_dir) / "detection_at_injected_rate.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def compute_pr_and_save(df, out_dir, dataset_name):
    os.makedirs(out_dir, exist_ok=True)
    summary_rows = []
    for split, g in df.items():
        scores = g["score"].values
        y = g["is_injected"].values.astype(int)
        if y.sum() == 0:
            print(f"Warning: no injected samples found for split={split}; skipping PR.")
            continue
        precision, recall, thresholds = precision_recall_curve(y, scores)
        ap = average_precision_score(y, scores)
        # save PR data
        prdf = pd.DataFrame({"precision": precision, "recall": recall})
        prcsv = Path(out_dir) / f"pr_data_{split}.csv"
        prdf.to_csv(prcsv, index=False)
        # plot
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.plot(recall, precision, label=f"AP={ap:.4f}")
        # annotate recall at injected fraction
        inj_frac = float(g["is_injected"].sum() / max(1, len(g)))
        ax.axvline(
            inj_frac, color="gray", linestyle="--", label=f"inj_frac={inj_frac:.3f}"
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"PR curve ({dataset_name} {split})")
        ax.legend()
        fig.tight_layout()
        figpath = Path(out_dir) / f"pr_curve_{split}.png"
        fig.savefig(figpath)
        plt.close(fig)
        summary_rows.append(
            {
                "dataset": dataset_name,
                "split": split,
                "ap": float(ap),
                "n_pos": int(y.sum()),
                "n_total": int(len(y)),
            }
        )

    summary_csv = Path(out_dir) / "pr_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    return summary_csv


def compute_distributions(df, out_dir, dataset_name):
    os.makedirs(out_dir, exist_ok=True)
    stats_rows = []
    for split, g in df.items():
        scores = g["score"].values
        is_inj = g["is_injected"].values.astype(bool)
        s_in = scores[is_inj]
        s_n = scores[~is_inj]

        # summary stats
        def summary(arr):
            return {
                "count": int(len(arr)),
                "mean": float(np.mean(arr)) if len(arr) else float("nan"),
                "median": float(np.median(arr)) if len(arr) else float("nan"),
                "std": float(np.std(arr)) if len(arr) else float("nan"),
                "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25))
                if len(arr)
                else float("nan"),
            }

        sum_in = summary(s_in)
        sum_n = summary(s_n)
        # statistical tests
        ks_p = None
        mw_p = None
        cohens_d = None
        if stats is not None and len(s_in) > 0 and len(s_n) > 0:
            try:
                ks_res = stats.ks_2samp(s_in, s_n)
                ks_p = float(ks_res.pvalue)
            except Exception:
                ks_p = None
            try:
                mw_res = stats.mannwhitneyu(s_in, s_n, alternative="two-sided")
                mw_p = float(mw_res.pvalue)
            except Exception:
                mw_p = None
            # Cohen's d
            try:
                pooled_std = np.sqrt(
                    (
                        (len(s_in) - 1) * np.var(s_in, ddof=1)
                        + (len(s_n) - 1) * np.var(s_n, ddof=1)
                    )
                    / (len(s_in) + len(s_n) - 2)
                )
                cohens_d = (
                    float((np.mean(s_in) - np.mean(s_n)) / pooled_std)
                    if pooled_std > 0
                    else None
                )
            except Exception:
                cohens_d = None

        stats_rows.append(
            {
                "dataset": dataset_name,
                "split": split,
                "n_injected": int(len(s_in)),
                "n_not_injected": int(len(s_n)),
                "injected_mean": sum_in["mean"],
                "not_injected_mean": sum_n["mean"],
                "injected_median": sum_in["median"],
                "not_injected_median": sum_n["median"],
                "injected_std": sum_in["std"],
                "not_injected_std": sum_n["std"],
                "ks_pvalue": ks_p,
                "mw_pvalue": mw_p,
                "cohens_d": cohens_d,
            }
        )

        # plot histogram + KDE (approx)
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        try:
            bins = 80
            ax.hist(s_n, bins=bins, density=True, alpha=0.6, label="not_injected")
            ax.hist(s_in, bins=bins, density=True, alpha=0.6, label="injected")
            ax.set_title(f"Score distributions ({dataset_name} {split})")
            ax.legend()
            fig.tight_layout()
            figpath = Path(out_dir) / f"score_distributions_{split}.png"
            fig.savefig(figpath)
            plt.close(fig)
        except Exception:
            pass

    stats_csv = Path(out_dir) / "score_distribution_stats.csv"
    pd.DataFrame(stats_rows).to_csv(stats_csv, index=False)
    return stats_csv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", default=None)
    args = p.parse_args()
    eval_dir = args.eval_dir
    data_dir = args.data_dir
    out_dir = args.out_dir or eval_dir
    dataset_name = Path(eval_dir).name

    print(f"Loading evaluation results from: {eval_dir}")
    entries = load_eval_results(eval_dir)
    print(f"Loaded {len(entries)} evaluation entries")
    if len(entries) > 0:
        print("sample entry keys:", list(entries[0].keys())[:10])

    # If entries are top-level split objects with a 'results' dict containing
    # `log_perplexity_values`, flatten here into per-trajectory entries.
    if all(isinstance(e, dict) and "split" in e and "results" in e for e in entries):
        flat = []
        for item in entries:
            res = item.get("results")
            if isinstance(res, dict) and "log_perplexity_values" in res:
                lp = res.get("log_perplexity_values", [])
                for i, v in enumerate(lp):
                    flat.append(
                        {
                            "row_index": i,
                            "log_perplexity_values": [v],
                            "split": item.get("split"),
                        }
                    )
        if flat:
            print(
                f"Flattened to {len(flat)} per-trajectory entries from split-level results"
            )
            entries = flat
    if not entries:
        print(f"No evaluation entries found in {eval_dir}. Exiting.")
        sys.exit(1)

    print(f"Loading injected indices from: {data_dir}")
    injected_map = load_injected_map(data_dir)
    if not injected_map:
        print(
            f"No injected indices found under {data_dir}. Proceeding but results will show zero positives."
        )

    print("Matching evaluation entries to injected indices...")
    tables = match_and_build_table(entries, injected_map)
    if not tables:
        print("No per-split tables created; exiting.")
        sys.exit(1)

    print("Computing detection metrics at injected fraction (top-13% & top-15%)...")
    det_csv = compute_detection_at_injected(tables, out_dir, dataset_name)
    print(f"Wrote detection CSV: {det_csv}")

    print("Computing PR curves and AP...")
    pr_csv = compute_pr_and_save(tables, out_dir, dataset_name)
    print(f"Wrote PR summary: {pr_csv}")

    print("Computing score distribution plots and stats...")
    stats_csv = compute_distributions(tables, out_dir, dataset_name)
    print(f"Wrote distribution stats: {stats_csv}")

    print("Done.")


if __name__ == "__main__":
    main()
