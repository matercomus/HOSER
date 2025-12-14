import json
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
    auc,
)


def find_csv_for_split(eval_dir: Path, dataset_name: str, split: str) -> Path:
    """Return the best guess CSV path used for evaluation for given dataset and split.

    Tries several locations in order:
      - data/<dataset>/<split>.csv
      - data/<dataset>_abnormal/<split>.csv
      - tools_eval_lmtad/<dataset>/<split>_sampled.csv
      - tools_eval_lmtad/<dataset>/<split>.csv
      - data/<dataset.replace('_abnormal','')>/<split>.csv
    """
    candidates = []
    candidates.append(Path("data") / dataset_name / f"{split}.csv")
    candidates.append(Path("data") / f"{dataset_name}_abnormal" / f"{split}.csv")
    candidates.append(Path("tools_eval_lmtad") / dataset_name / f"{split}_sampled.csv")
    candidates.append(Path("tools_eval_lmtad") / dataset_name / f"{split}.csv")
    if dataset_name.endswith("_abnormal"):
        base = dataset_name.replace("_abnormal", "")
        candidates.append(Path("data") / base / f"{split}.csv")

    for p in candidates:
        if p.exists():
            return p
    return None


def evaluate_dataset_eval_dirs(root: Path = Path("tools_eval_lmtad")):
    root = Path(root)
    summary_rows = []
    per_dataset = defaultdict(list)

    for eval_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        eval_file = eval_dir / "evaluation_results.json"
        if not eval_file.exists():
            continue
        dataset_name = eval_dir.name
        agg = json.load(open(eval_file))
        out_rows = []

        for split, res in agg.items():
            outlier_labels = res.get("outlier_labels", [])
            reported_rate = res.get("outlier_rate")
            # load per-row scores if available
            scores = res.get("log_perplexity_values") or res.get("scores") or []

            csv_path = find_csv_for_split(eval_dir, dataset_name, split)
            if csv_path is None:
                print(f"Skipping {dataset_name}/{split}: no CSV found")
                continue

            # read ground-truth abnormality_info column if present
            gt = []
            with open(csv_path, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    ai = row.get("abnormality_info", "")
                    gt.append(1 if ai and ai != "normal" else 0)

            # Truncate to shortest (evaluation may have used a sampled CSV)
            n_csv = len(gt)
            n_eval = len(outlier_labels)
            n = min(n_csv, n_eval)
            if n == 0:
                print(f"No comparable rows for {dataset_name}/{split}")
                continue

            gt = gt[:n]
            pred = outlier_labels[:n]

            # Porto-style 3-sigma detector (uses non-outlier rows to set threshold)
            porto_metrics = {}
            if scores and len(scores) >= n:
                scores_arr = np.array(scores[:n], dtype=float)
                # Identify non-outlier rows from GT if present; otherwise assume lowest 95% are non-outliers
                non_outlier_mask = np.array([g == 0 for g in gt])
                if non_outlier_mask.sum() == 0:
                    # fallback: take lower 95% as non-outlier set
                    th_quant = np.quantile(scores_arr, 0.95)
                    non_outlier_mask = scores_arr <= th_quant

                non_outlier_scores = scores_arr[non_outlier_mask]
                if non_outlier_scores.size > 1:
                    std_above = 3.0
                    std = non_outlier_scores.std(ddof=0) * std_above
                    threshold_porto = non_outlier_scores.mean() + std
                else:
                    threshold_porto = float(np.quantile(scores_arr, 0.95))

                # detected labels (no rounding)
                detected_porto = (scores_arr > threshold_porto).astype(int)
                n_gt = int((scores_arr > threshold_porto).sum())
                n_eq = int((scores_arr == threshold_porto).sum())

                tp_p = int(
                    sum(1 for a, b in zip(gt, detected_porto) if a == 1 and b == 1)
                )
                fp_p = int(
                    sum(1 for a, b in zip(gt, detected_porto) if a == 0 and b == 1)
                )
                fn_p = int(
                    sum(1 for a, b in zip(gt, detected_porto) if a == 1 and b == 0)
                )
                tn_p = int(
                    sum(1 for a, b in zip(gt, detected_porto) if a == 0 and b == 0)
                )

                prec_p = precision_score(gt, detected_porto, zero_division=0)
                rec_p = recall_score(gt, detected_porto, zero_division=0)
                f1_p = f1_score(gt, detected_porto, zero_division=0)
                acc_p = accuracy_score(gt, detected_porto)

                # average precision and PR AUC using continuous scores
                try:
                    avg_prec = average_precision_score(gt, scores_arr)
                    precisions, recalls, _ = precision_recall_curve(gt, scores_arr)
                    pr_auc_v = auc(recalls, precisions)
                except Exception:
                    avg_prec = None
                    pr_auc_v = None

                porto_metrics = {
                    "porto_threshold": float(threshold_porto),
                    "porto_n_gt": n_gt,
                    "porto_n_eq": n_eq,
                    "porto_tp": tp_p,
                    "porto_fp": fp_p,
                    "porto_fn": fn_p,
                    "porto_tn": tn_p,
                    "porto_precision": float(prec_p),
                    "porto_recall": float(rec_p),
                    "porto_f1": float(f1_p),
                    "porto_accuracy": float(acc_p),
                    "porto_average_precision": None
                    if avg_prec is None
                    else float(avg_prec),
                    "porto_pr_auc": None if pr_auc_v is None else float(pr_auc_v),
                }
            else:
                porto_metrics = {}

            tp = sum(1 for a, b in zip(gt, pred) if a == 1 and b == 1)
            fp = sum(1 for a, b in zip(gt, pred) if a == 0 and b == 1)
            fn = sum(1 for a, b in zip(gt, pred) if a == 1 and b == 0)
            tn = sum(1 for a, b in zip(gt, pred) if a == 0 and b == 0)

            prec = tp / (tp + fp) if (tp + fp) > 0 else None
            rec = tp / (tp + fn) if (tp + fn) > 0 else None
            f1 = (
                (2 * prec * rec / (prec + rec))
                if (prec and rec and (prec + rec) > 0)
                else None
            )

            injected_rate = sum(gt) / n

            row = {
                "dataset": dataset_name,
                "split": split,
                "csv_path": str(csv_path),
                "n_csv": n_csv,
                "n_eval_labels": n_eval,
                "n_compared": n,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "injected_rate": injected_rate,
                "reported_outlier_rate": reported_rate,
            }

            # merge porto metrics if computed
            if porto_metrics:
                row.update(porto_metrics)

            out_rows.append(row)
            per_dataset[dataset_name].append(row)
            summary_rows.append(row)

            print(
                f"{dataset_name}/{split}: compared={n}, TP={tp}, FP={fp}, FN={fn}, TN={tn}, precision={prec}, recall={rec}, injected={injected_rate:.4f}, reported={reported_rate}"
            )

        # write per-dataset CSV summary
        out_file = eval_dir / "confusion_summary.csv"
        if out_rows:
            keys = list(out_rows[0].keys())
            with open(out_file, "w", newline="") as cf:
                writer = csv.DictWriter(cf, fieldnames=keys)
                writer.writeheader()
                for r in out_rows:
                    writer.writerow(r)
            print(f"Wrote {out_file}")

    # write global summary
    all_file = root / "confusion_summary_all.csv"
    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(all_file, "w", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=keys)
            writer.writeheader()
            for r in summary_rows:
                writer.writerow(r)
        print(f"Wrote {all_file}")


if __name__ == "__main__":
    evaluate_dataset_eval_dirs(Path("tools_eval_lmtad"))
