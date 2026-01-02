#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run LM-TAD decision benchmark (overnight-friendly).

Defaults match the repo conventions; override via flags or env vars.

Required:
  --ckpt PATH            Path to LM-TAD checkpoint

Optional:
  --name NAME            Run name (default: lmtad_decision_bench_overnight_<timestamp>)
  --out-dir DIR          Output root (default: research_runs/_benchmarks)
  --baseline-eval DIR    Baseline eval dir containing baseline_eval.json (default: tools_eval_lmtad/Beijing)
  --baseline-data DIR    Baseline data dir (roadmap fallback) (default: data/Beijing)
  --baseline-split SPLIT Baseline split in baseline_eval.json (default: train)
  --split SPLIT          Target split (default: train)
  --device DEV           Torch device (default: cuda:0)
  --batch-size N         Batch size (default: 128)
  --q LIST               Comma-separated quantiles (default: 0.50,...,0.99)
  --targets-default      Use default Beijing detectable datasets (3 variants)
  --target-dirs LIST     Comma-separated target dataset dirs

Examples:
  scripts/run_lmtad_decision_benchmark_overnight.sh --ckpt /path/to/ckpt.pt --targets-default

  scripts/run_lmtad_decision_benchmark_overnight.sh \
    --ckpt /path/to/ckpt.pt \
    --target-dirs data/Beijing_abnormal_3_detectable,data/Beijing_abnormal_3_detectable_dr \
    --device cuda:1
EOF
}

# Defaults
NAME="lmtad_decision_bench_overnight_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="research_runs/_benchmarks"
BASELINE_EVAL="tools_eval_lmtad/Beijing"
BASELINE_DATA_DIR="data/Beijing"
BASELINE_SPLIT="train"
SPLIT="train"
DEVICE="cuda:0"
BATCH_SIZE="128"
Q_LIST="0.50,0.60,0.70,0.80,0.85,0.90,0.92,0.94,0.95,0.96,0.97,0.98,0.99"
TARGETS_DEFAULT="false"
TARGET_DIRS=""

CKPT="/home/mka299/LMTAD/code/results/LMTAD/beijing_hoser_reference/run_20250928_202718/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/ckpt_best.pt"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --name)
      NAME="$2"; shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"; shift 2
      ;;
    --baseline-eval)
      BASELINE_EVAL="$2"; shift 2
      ;;
    --baseline-data)
      BASELINE_DATA_DIR="$2"; shift 2
      ;;
    --baseline-split)
      BASELINE_SPLIT="$2"; shift 2
      ;;
    --split)
      SPLIT="$2"; shift 2
      ;;
    --device)
      DEVICE="$2"; shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2
      ;;
    --q)
      Q_LIST="$2"; shift 2
      ;;
    --targets-default)
      TARGETS_DEFAULT="true"; shift
      ;;
    --target-dirs)
      TARGET_DIRS="$2"; shift 2
      ;;
    --ckpt)
      CKPT="$2"; shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$CKPT" ]]; then
  echo "Missing required --ckpt" >&2
  usage >&2
  exit 2
fi

if [[ "$TARGETS_DEFAULT" != "true" && -z "$TARGET_DIRS" ]]; then
  echo "Provide --targets-default or --target-dirs" >&2
  usage >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
LOG_PATH="$OUT_DIR/${NAME}.log"

CMD=(
  uv run python tools/run_lmtad_decision_benchmark.py
  --name "$NAME"
  --out-dir "$OUT_DIR"
  --baseline-eval "$BASELINE_EVAL"
  --baseline-data-dir "$BASELINE_DATA_DIR"
  --baseline-split "$BASELINE_SPLIT"
  --split "$SPLIT"
  --device "$DEVICE"
  --batch-size "$BATCH_SIZE"
  --q "$Q_LIST"
  --ckpt "$CKPT"
)

if [[ "$TARGETS_DEFAULT" == "true" ]]; then
  CMD+=(--target-data-dirs-default)
else
  CMD+=(--target-data-dirs "$TARGET_DIRS")
fi

echo "Logging to: $LOG_PATH" >&2
echo "Command: ${CMD[*]}" >&2

# tmux-friendly: keep foreground output while also writing to a log.
"${CMD[@]}" 2>&1 | tee "$LOG_PATH"
