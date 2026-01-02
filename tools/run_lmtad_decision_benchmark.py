#!/usr/bin/env python3
"""Benchmark LM-TAD decision rules across multiple abnormal datasets.

What this does (one script):
- Builds balanced test sets for each target dataset (all abnormal + matched normals)
- Runs LM-TAD scoring once per target
- Evaluates two decision methods:
  1) Baseline-quantile thresholding at q values (fixed baseline FPR)
  2) Top-k selection (alert-budget), evaluated at k matched to each q via k=ceil((1-q)*N)
- Computes overall + per-type metrics and writes plots + a Markdown report

Outputs (under --out-dir/--name):
- balanced_data/<dataset>/<split>.csv
- eval/<dataset>/evaluation_results.json
- analysis/<dataset>/metrics.json
- analysis/<dataset>/report.md
- analysis/<dataset>/plots/*.png
- analysis/summary.json (across datasets)
- analysis/summary.md

Example
uv run python tools/run_lmtad_decision_benchmark.py \
  --name bj_abn3_detectable_compare \
  --baseline-eval tools_eval_lmtad/Beijing \
  --baseline-data-dir data/Beijing \
  --target-data-dirs data/Beijing_abnormal_3_detectable,data/Beijing_abnormal_3_detectable_dr,data/Beijing_abnormal_3_detectable_route_switch \
  --split train \
  --normal-per-abnormal 1 \
  --ckpt /path/to/ckpt_best.pt \
  --lmtad-repo /home/mka299/LMTAD \
  --device cuda:0 \
  --batch-size 128 \
  --q 0.90,0.95,0.99 \
  --out-dir research_runs/_benchmarks
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# Fix `import tools.*` when executed as `python tools/...py`
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_NULL_LIKE = {"", "nan", "none", "null"}


def _is_abnormal(raw: str) -> bool:
    s = str(raw or "").strip().lower()
    if s in _NULL_LIKE or s == "normal":
        return False
    return True


def _parse_type(raw: str) -> str:
    if not _is_abnormal(raw):
        return "normal"
    try:
        obj = ast.literal_eval(raw)
        if isinstance(obj, dict) and "type" in obj:
            return str(obj["type"])
    except Exception:
        pass
    s = str(raw).lower()
    if "route_switch" in s:
        return "route_switch"
    if "detour" in s:
        return "detour"
    return "abnormal"


def _read_labels(csv_path: Path, *, label_col: str = "abnormality_info") -> Tuple[np.ndarray, List[str]]:
    import csv

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or label_col not in reader.fieldnames:
            raise ValueError(f"CSV missing '{label_col}': {csv_path}")
        labels: List[bool] = []
        raw: List[str] = []
        for row in reader:
            v = (row.get(label_col) or "").strip()
            labels.append(_is_abnormal(v))
            raw.append(v)
    return np.asarray(labels, dtype=bool), raw


def _load_eval_scores(eval_dir: Path, split: str) -> np.ndarray:
    payload = json.loads((eval_dir / "evaluation_results.json").read_text(encoding="utf-8"))
    scores = np.asarray(payload[split]["log_perplexity_values"], dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError(f"No finite scores for split '{split}'")
    return scores


def _load_baseline_scores(baseline_eval: Path, split: str) -> np.ndarray:
    path = baseline_eval
    if path.is_dir():
        cand = path / "baseline_eval.json"
        if cand.exists():
            path = cand
        else:
            raise FileNotFoundError(f"Missing baseline_eval.json in {baseline_eval}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if split not in payload:
        raise ValueError(f"Split '{split}' not found in baseline eval: {path}")

    scores = np.asarray(payload[split].get("log_perplexity_values", []), dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError(f"No finite baseline scores for split '{split}'")
    return scores


def _confusion(scores: np.ndarray, labels: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    y = labels.astype(bool)
    p = pred.astype(bool)
    tp = int(np.logical_and(p, y).sum())
    fp = int(np.logical_and(p, ~y).sum())
    fn = int(np.logical_and(~p, y).sum())
    tn = int(np.logical_and(~p, ~y).sum())
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
    }


def _auroc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    y = labels.astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(-scores)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    tpr = tp / float(n_pos)
    fpr = fp / float(n_neg)
    tpr = np.concatenate(([0.0], tpr))
    fpr = np.concatenate(([0.0], fpr))
    return float(np.trapezoid(tpr, fpr))


def _auprc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    y = labels.astype(bool)
    n_pos = int(y.sum())
    if n_pos == 0:
        return None

    order = np.argsort(-scores)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    precision = tp / (tp + fp)
    return float(precision[y_sorted].sum() / float(n_pos))


def _topk_pred(scores: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
    k = int(max(0, min(int(k), int(scores.size))))
    if k == 0:
        return np.zeros_like(scores, dtype=bool), float("inf")
    # kth largest threshold.
    order = np.argsort(-scores)
    cutoff_idx = order[k - 1]
    cutoff = float(scores[cutoff_idx])
    pred = scores >= cutoff
    # If ties expand slightly beyond k, that's OK; report effective rate.
    return pred, cutoff


def _plot_hist(scores: np.ndarray, labels: np.ndarray, vlines: Dict[str, float], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pos = scores[labels]
    neg = scores[~labels]

    plt.figure(figsize=(8, 4.5))
    plt.hist(neg, bins=60, alpha=0.55, label=f"normal (n={neg.size})")
    plt.hist(pos, bins=60, alpha=0.55, label=f"abnormal (n={pos.size})")
    for name, x in vlines.items():
        plt.axvline(float(x), linestyle="--", linewidth=1.4, label=name)
    plt.title("LM-TAD log-perplexity (balanced)")
    plt.xlabel("log perplexity")
    plt.ylabel("count")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_type_box(scores: np.ndarray, types: List[str], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys = [k for k in sorted(set(types))]
    data = [scores[np.asarray([t == k for t in types], dtype=bool)] for k in keys]

    plt.figure(figsize=(7, 4.5))
    plt.boxplot(data, labels=keys, showfliers=False)
    plt.title("Score by abnormality type")
    plt.ylabel("log perplexity")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _basename(dataset_dir: Path) -> str:
    return dataset_dir.name


def _resolve_roadmap(balanced_dir: Path, target_data_dir: Path, baseline_data_dir: Optional[Path], roadmap_arg: Optional[Path]) -> Path:
    if roadmap_arg is not None:
        roadmap = Path(roadmap_arg)
        if not roadmap.exists():
            raise FileNotFoundError(f"Roadmap not found: {roadmap}")
        return roadmap

    candidates = [balanced_dir / "roadmap.geo", target_data_dir / "roadmap.geo"]
    if baseline_data_dir is not None:
        candidates.append(baseline_data_dir / "roadmap.geo")
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Roadmap file not found. Provide --roadmap or ensure roadmap.geo exists.")


def _run_balancer(*, target_data_dir: Path, split: str, out_dir: Path, normal_per_abnormal: int, length_bucket: int, seed: int,
                  allow_replacement: bool, copy_roadmaps: bool) -> Path:
    from tools.make_balanced_eval_dataset import main as balance_main

    args = [
        "--source-dataset-dir",
        str(target_data_dir),
        "--split",
        str(split),
        "--out-dir",
        str(out_dir),
        "--normal-per-abnormal",
        str(int(normal_per_abnormal)),
        "--length-bucket",
        str(int(length_bucket)),
        "--seed",
        str(int(seed)),
    ]
    if allow_replacement:
        args.append("--allow-replacement")
    if copy_roadmaps:
        args.append("--copy-roadmaps")

    old = sys.argv
    try:
        sys.argv = ["make_balanced_eval_dataset.py", *args]
        balance_main()
    finally:
        sys.argv = old

    csv_path = out_dir / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Balanced CSV not created: {csv_path}")
    return csv_path


def _run_eval(*, data_dir: Path, output_dir: Path, roadmap: Path, ckpt: Path, lmtad_repo: Path, device: str, batch_size: int,
             split: str, baseline_eval: Path, baseline_quantile: float, baseline_split: str) -> None:
    from tools import evaluate_dataset_with_lmtad as eval_mod

    eval_mod.evaluate_splits(
        data_dir=data_dir,
        roadmap_file=roadmap,
        lmtad_ckpt=ckpt,
        lmtad_repo=lmtad_repo,
        device=device,
        batch_size=int(batch_size),
        splits=[split],
        output_dir=output_dir,
        sample_frac=1.0,
        sample_seed=42,
        baseline_eval=baseline_eval,
        baseline_quantile=float(baseline_quantile),
        baseline_split=str(baseline_split),
        write_baseline=False,
    )


def _analyze_decisions(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    types: List[str],
    baseline_scores: np.ndarray,
    q_list: Sequence[float],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["n"] = int(labels.size)
    out["n_pos"] = int(labels.sum())
    out["n_neg"] = int((~labels).sum())
    out["pos_fraction"] = float(labels.mean()) if labels.size else 0.0
    out["auroc"] = _auroc(scores, labels)
    out["auprc"] = _auprc(scores, labels)

    # Per-type counts.
    type_counts: Dict[str, int] = {}
    for t in types:
        type_counts[t] = type_counts.get(t, 0) + 1
    out["type_counts"] = type_counts

    per_q: Dict[str, Any] = {}
    per_q_topk: Dict[str, Any] = {}

    for q in q_list:
        thr = float(np.quantile(baseline_scores, float(q)))
        pred_q = scores >= thr
        m_q = _confusion(scores, labels, pred_q)
        m_q["threshold"] = thr
        m_q["method"] = "baseline_quantile"
        m_q["q"] = float(q)
        m_q["flag_rate"] = float(pred_q.mean())

        # per-type recall
        by_type: Dict[str, Any] = {}
        for t in sorted(set(types)):
            idx = np.asarray([tt == t for tt in types], dtype=bool)
            if not idx.any():
                continue
            y_t = labels[idx]
            p_t = pred_q[idx]
            tp = int(np.logical_and(p_t, y_t).sum())
            fn = int(np.logical_and(~p_t, y_t).sum())
            rec = float(tp / (tp + fn)) if (tp + fn) else 0.0
            by_type[t] = {"n": int(idx.sum()), "n_pos": int(y_t.sum()), "recall": rec}
        m_q["by_type"] = by_type
        per_q[f"q={float(q):.2f}"] = m_q

        # Top-k at matched alert volume: k = ceil((1-q) * N)
        k = int(np.ceil((1.0 - float(q)) * float(scores.size)))
        pred_k, cutoff = _topk_pred(scores, k)
        m_k = _confusion(scores, labels, pred_k)
        m_k["method"] = "topk_matched_to_q"
        m_k["q_matched"] = float(q)
        m_k["k"] = int(k)
        m_k["cutoff"] = float(cutoff)
        m_k["flag_rate"] = float(pred_k.mean())

        by_type_k: Dict[str, Any] = {}
        for t in sorted(set(types)):
            idx = np.asarray([tt == t for tt in types], dtype=bool)
            if not idx.any():
                continue
            y_t = labels[idx]
            p_t = pred_k[idx]
            tp = int(np.logical_and(p_t, y_t).sum())
            fn = int(np.logical_and(~p_t, y_t).sum())
            rec = float(tp / (tp + fn)) if (tp + fn) else 0.0
            by_type_k[t] = {"n": int(idx.sum()), "n_pos": int(y_t.sum()), "recall": rec}
        m_k["by_type"] = by_type_k
        per_q_topk[f"q={float(q):.2f}"] = m_k

    out["baseline_quantile"] = per_q
    out["topk_matched"] = per_q_topk
    return out


def _render_dataset_report(
    *,
    dataset_name: str,
    split: str,
    metrics: Dict[str, Any],
    plots_rel: Dict[str, str],
) -> str:
    lines: List[str] = []
    lines.append(f"# LM-TAD decision benchmark: `{dataset_name}` ({split})")
    lines.append("")
    lines.append(f"- N={metrics['n']} pos={metrics['n_pos']} neg={metrics['n_neg']} pos_fraction={metrics['pos_fraction']:.2%}")
    if metrics.get("auroc") is not None:
        lines.append(f"- AUROC={float(metrics['auroc']):.4f}")
    if metrics.get("auprc") is not None:
        lines.append(f"- AUPRC={float(metrics['auprc']):.4f}")
    lines.append("")

    lines.append("## Plots")
    lines.append("")
    for title, rel in plots_rel.items():
        lines.append(f"![{title}]({rel})")
        lines.append("")

    lines.append("## Per-type composition")
    lines.append("")
    for k in sorted(metrics.get("type_counts", {}).keys()):
        lines.append(f"- `{k}`: {metrics['type_counts'][k]}")

    lines.append("")
    lines.append("## Decision rules")
    lines.append("")

    lines.append("### Baseline-quantile thresholds")
    lines.append("")
    lines.append("| q | thr | flag_rate | recall | precision | FPR | TP | FP | FN | TN |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in sorted(metrics["baseline_quantile"].keys()):
        m = metrics["baseline_quantile"][key]
        lines.append(
            f"| {float(m['q']):.2f} | {float(m['threshold']):.6f} | {float(m['flag_rate']):.3f} | {float(m['recall']):.3f} | {float(m['precision']):.3f} | {float(m['fpr']):.3f} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |"
        )

    lines.append("")
    lines.append("### Top-k matched to q (same alert volume)")
    lines.append("")
    lines.append("| q_match | k | cutoff | flag_rate | recall | precision | FPR | TP | FP | FN | TN |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in sorted(metrics["topk_matched"].keys()):
        m = metrics["topk_matched"][key]
        lines.append(
            f"| {float(m['q_matched']):.2f} | {int(m['k'])} | {float(m['cutoff']):.6f} | {float(m['flag_rate']):.3f} | {float(m['recall']):.3f} | {float(m['precision']):.3f} | {float(m['fpr']):.3f} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |"
        )

    lines.append("")
    lines.append("## Interpretation (LM-TAD vs method)")
    lines.append("")
    lines.append(
        "- AUROC/AUPRC capture score separation (ranking quality).\n"
        "- q-thresholding is a fixed-FPR method anchored to the baseline distribution.\n"
        "- top-k is an alert-budget method (always flags k items).\n"
        "Compare matched-q top-k vs q-thresholding to see whether the *decision rule* is limiting recall/precision even when separation exists."
    )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark q-threshold vs top-k for LM-TAD across datasets")
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)

    parser.add_argument("--baseline-eval", type=Path, required=True)
    parser.add_argument("--baseline-data-dir", type=Path, default=None)
    parser.add_argument("--baseline-split", type=str, default="train")

    parser.add_argument(
        "--target-data-dirs",
        type=str,
        required=False,
        default=None,
        help="Comma-separated dataset dirs to evaluate",
    )
    parser.add_argument(
        "--target-data-dirs-default",
        action="store_true",
        help="Use default Beijing detectable datasets (3 variants) under data/",
    )

    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--normal-per-abnormal", type=int, default=1)
    parser.add_argument("--length-bucket", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-replacement", action="store_true")
    parser.add_argument("--copy-roadmaps", action="store_true")
    parser.add_argument("--roadmap", type=Path, default=None)

    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--lmtad-repo", type=Path, default=Path("/home/mka299/LMTAD"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)

    parser.add_argument("--q", type=str, default="0.90,0.95,0.99")

    args = parser.parse_args()

    name = str(args.name)
    out_root = Path(args.out_dir) / name

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(f"LM-TAD checkpoint not found: {ckpt}")
    lmtad_repo = Path(args.lmtad_repo)
    if not lmtad_repo.exists():
        raise FileNotFoundError(f"LM-TAD repo not found: {lmtad_repo}")
    baseline_eval = Path(args.baseline_eval)
    if not baseline_eval.exists():
        raise FileNotFoundError(f"Baseline eval not found: {baseline_eval}")

    split = str(args.split)
    baseline_split = str(args.baseline_split)

    if args.target_data_dirs_default:
        targets = [
            Path("data/Beijing_abnormal_3_detectable"),
            Path("data/Beijing_abnormal_3_detectable_dr"),
            Path("data/Beijing_abnormal_3_detectable_route_switch"),
        ]
    else:
        if not args.target_data_dirs:
            raise ValueError("Provide --target-data-dirs or use --target-data-dirs-default")
        targets = [Path(x.strip()) for x in str(args.target_data_dirs).split(",") if x.strip()]

    for t in targets:
        if not t.exists():
            raise FileNotFoundError(f"Target dataset dir not found: {t}")
        if not (t / f"{split}.csv").exists():
            raise FileNotFoundError(f"Target split CSV not found: {t / f'{split}.csv'}")

    q_list = [float(x.strip()) for x in str(args.q).split(",") if x.strip()]
    if not q_list:
        raise ValueError("No q values provided")

    baseline_scores = _load_baseline_scores(baseline_eval, baseline_split)

    summary: Dict[str, Any] = {
        "name": name,
        "split": split,
        "baseline_eval": str(baseline_eval),
        "baseline_split": baseline_split,
        "q": q_list,
        "datasets": {},
    }

    for target_dir in targets:
        ds_name = _basename(target_dir)

        balanced_dir = out_root / "balanced_data" / ds_name
        eval_dir = out_root / "eval" / ds_name
        analysis_dir = out_root / "analysis" / ds_name
        plots_dir = analysis_dir / "plots"

        csv_path = _run_balancer(
            target_data_dir=target_dir,
            split=split,
            out_dir=balanced_dir,
            normal_per_abnormal=int(args.normal_per_abnormal),
            length_bucket=int(args.length_bucket),
            seed=int(args.seed),
            allow_replacement=bool(args.allow_replacement),
            copy_roadmaps=bool(args.copy_roadmaps),
        )

        roadmap = _resolve_roadmap(
            balanced_dir=balanced_dir,
            target_data_dir=target_dir,
            baseline_data_dir=Path(args.baseline_data_dir) if args.baseline_data_dir is not None else None,
            roadmap_arg=Path(args.roadmap) if args.roadmap is not None else None,
        )

        _run_eval(
            data_dir=balanced_dir,
            output_dir=eval_dir,
            roadmap=roadmap,
            ckpt=ckpt,
            lmtad_repo=lmtad_repo,
            device=str(args.device),
            batch_size=int(args.batch_size),
            split=split,
            baseline_eval=baseline_eval,
            baseline_quantile=float(max(q_list)),
            baseline_split=baseline_split,
        )

        scores = _load_eval_scores(eval_dir, split)
        labels, raw = _read_labels(csv_path)
        if scores.size != labels.size:
            raise ValueError(
                f"Length mismatch for {ds_name}: scores={scores.size} labels={labels.size}. "
                "Ensure evaluation used the same balanced CSV."
            )
        types = [_parse_type(x) for x in raw]

        metrics = _analyze_decisions(
            scores=scores,
            labels=labels,
            types=types,
            baseline_scores=baseline_scores,
            q_list=q_list,
        )

        # Plots: hist + type boxplot. Add a couple reference vlines.
        vlines: Dict[str, float] = {}
        for q in q_list:
            vlines[f"thr@q={q:.2f}"] = float(np.quantile(baseline_scores, q))
        # Also top-k cutoff for the middle q (if exists)
        q_mid = q_list[len(q_list) // 2]
        k_mid = int(np.ceil((1.0 - float(q_mid)) * float(scores.size)))
        pred_mid, cutoff_mid = _topk_pred(scores, k_mid)
        vlines[f"topk@q={q_mid:.2f} cutoff"] = float(cutoff_mid)

        hist_path = plots_dir / "score_hist.png"
        box_path = plots_dir / "score_by_type_box.png"
        _plot_hist(scores, labels, vlines, hist_path)
        _plot_type_box(scores, types, box_path)

        plots_rel = {
            "Score histogram": str(hist_path.relative_to(analysis_dir)),
            "Score by type (box)": str(box_path.relative_to(analysis_dir)),
        }

        report_md = analysis_dir / "report.md"
        _write_md(
            report_md,
            _render_dataset_report(dataset_name=ds_name, split=split, metrics=metrics, plots_rel=plots_rel),
        )

        metrics_path = analysis_dir / "metrics.json"
        _write_json(metrics_path, metrics)

        summary["datasets"][ds_name] = {
            "target_data_dir": str(target_dir),
            "balanced_csv": str(csv_path),
            "eval_dir": str(eval_dir),
            "analysis_dir": str(analysis_dir),
            "metrics": metrics,
        }

    # Write aggregate summary
    summary_json = out_root / "analysis" / "summary.json"
    _write_json(summary_json, summary)

    # Summary markdown (links)
    lines: List[str] = []
    lines.append(f"# LM-TAD benchmark summary: `{name}`")
    lines.append("")
    lines.append(f"Split: `{split}`")
    lines.append("")
    for ds_name in sorted(summary["datasets"].keys()):
        entry = summary["datasets"][ds_name]
        auroc = entry["metrics"].get("auroc")
        auprc = entry["metrics"].get("auprc")
        lines.append(f"- `{ds_name}`: AUROC={auroc:.4f} AUPRC={auprc:.4f} report=`{Path(entry['analysis_dir'])/'report.md'}`")
    lines.append("")
    lines.append("Use each per-dataset report for per-type breakdown and q vs top-k comparison.")

    summary_md = out_root / "analysis" / "summary.md"
    _write_md(summary_md, "\n".join(lines) + "\n")

    print(str(out_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
