#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

check_writable_dir() {
  # Best-effort check that a directory is writable by the current user.
  # This catches common issues like UID mismatch in containers or root-squash.
  local d="$1"
  mkdir -p "$d" 2>/dev/null || die "cannot create directory: $d"
  local probe="$d/.perm_probe_$$"
  : >"$probe" 2>/dev/null || die "directory not writable: $d (uid=$(id -u), gid=$(id -g))"
  rm -f "$probe" 2>/dev/null || true
}

# Non-interactive scripts do not source ~/.bashrc, so they won't pick up the
# user's `uv()` wrapper. Ensure uv uses the per-project venv under /local.
if [[ -z "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
  root_real="$(readlink -f "$ROOT_DIR")"
  hash="$(printf '%s' "$root_real" | sha1sum | awk '{print substr($1,1,8)}')"
  name="$(basename "$root_real")-$hash"
  local_user="${USER:-$(id -un 2>/dev/null || echo mka299)}"
  envdir="/local/data/${local_user}/uv/venvs/$name"
  check_writable_dir "$envdir"
  export UV_PROJECT_ENVIRONMENT="$envdir"
fi

usage() {
  cat <<'EOF'
Run LM-TAD decision benchmark (overnight-friendly).

Defaults match the repo conventions; override via flags or env vars.

Optional:
  --ckpt PATH            Override default checkpoint for both jobs (default: use checkpoint paths set in this script)
  --name NAME            Run name prefix (default: lmtad_decision_bench_overnight_<timestamp>)
  --out-dir DIR          Output root (default: research_runs/_benchmarks)
  --baseline-split SPLIT Baseline split in baseline_eval.json (default: train)
  --split SPLIT          Target split (default: train)
  --device DEV           Torch device (default: cuda:0)
  --batch-size N         Batch size (default: 128)
  --q LIST               Comma-separated quantiles (default: 0.50,...,0.99)

  # Multi-job convenience
  --preset NAME          Preset runner. Supported: per-type-detour
  --jobs LIST            Comma-separated jobs to run: beijing,porto (default: derived from preset)

  # Per-job dataset/baseline overrides
  --beijing-target-dir DIR   (default: data/_per_type/Beijing_per_type_detour)
  --porto-target-dir DIR     (default: data/_per_type/porto_hoser_per_type_detour)
  --beijing-ckpt PATH        Override checkpoint for the Beijing job (default: --ckpt)
  --porto-ckpt PATH          Override checkpoint for the Porto job (default: --ckpt)
  --beijing-baseline-eval DIR (default: tools_eval_lmtad/Beijing)
  --beijing-baseline-data DIR (default: data/Beijing)
  --porto-baseline-eval DIR   (default: tools_eval_lmtad/porto_hoser)
  --porto-baseline-data DIR   (default: data/porto_hoser)

  # Baseline creation
  --no-auto-baseline     Do not auto-generate missing baseline_eval.json

  # tmux
  --tmux-session NAME    Run jobs inside a tmux session (one window per job). If it exists, reuse it.
                         If you're already inside tmux and don't want new windows, omit this flag.

Examples:
  # Run both per-type detour benchmarks (Beijing + Porto)
  scripts/run_lmtad_decision_benchmark_overnight.sh --preset per-type-detour

  # Same, but in tmux (detach, disconnect, re-attach later)
  scripts/run_lmtad_decision_benchmark_overnight.sh --preset per-type-detour --tmux-session lmtad_detour

  # Run only Porto per-type detour
  scripts/run_lmtad_decision_benchmark_overnight.sh \
    --jobs porto \
    --porto-target-dir data/_per_type/porto_hoser_per_type_detour

  # Custom targets in a single job
  scripts/run_lmtad_decision_benchmark_overnight.sh \
    --jobs beijing \
    --beijing-target-dir data/Beijing_abnormal_3_detectable \
    --device cuda:1
EOF
}

# Defaults
NAME="lmtad_decision_bench_overnight_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="research_runs/_benchmarks"
BASELINE_SPLIT="train"
SPLIT="train"
DEVICE="cuda:0"
BATCH_SIZE="128"
Q_LIST="0.50,0.60,0.70,0.80,0.85,0.90,0.92,0.94,0.95,0.96,0.97,0.98,0.99"
PRESET=""
JOBS=""

BEIJING_TARGET_DIR="data/_per_type/Beijing_per_type_detour"
PORTO_TARGET_DIR="data/_per_type/porto_hoser_per_type_detour"

BEIJING_BASELINE_EVAL="tools_eval_lmtad/Beijing"
BEIJING_BASELINE_DATA_DIR="data/Beijing"

PORTO_BASELINE_EVAL="tools_eval_lmtad/porto_hoser"
PORTO_BASELINE_DATA_DIR="data/porto_hoser"

AUTO_BASELINE="true"
TMUX_SESSION=""

# Optional override for both jobs.
CKPT=""
BEIJING_CKPT="/home/mka299/LMTAD/code/results/LMTAD/beijing_hoser_reference/run_20250928_202718/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/ckpt_best.pt"
PORTO_CKPT="/home/mka299/LMTAD/code/results/LMTAD/porto_hoser/run_20251010_212829/outlier_False/n_layer_8_n_head_12_n_embd_768_lr_0.0003_integer_poe_False/ckpt_best.pt"

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
    --preset)
      PRESET="$2"; shift 2
      ;;
    --jobs)
      JOBS="$2"; shift 2
      ;;
    --beijing-target-dir)
      BEIJING_TARGET_DIR="$2"; shift 2
      ;;
    --porto-target-dir)
      PORTO_TARGET_DIR="$2"; shift 2
      ;;
    --beijing-ckpt)
      BEIJING_CKPT="$2"; shift 2
      ;;
    --porto-ckpt)
      PORTO_CKPT="$2"; shift 2
      ;;
    --beijing-baseline-eval)
      BEIJING_BASELINE_EVAL="$2"; shift 2
      ;;
    --beijing-baseline-data)
      BEIJING_BASELINE_DATA_DIR="$2"; shift 2
      ;;
    --porto-baseline-eval)
      PORTO_BASELINE_EVAL="$2"; shift 2
      ;;
    --porto-baseline-data)
      PORTO_BASELINE_DATA_DIR="$2"; shift 2
      ;;
    --no-auto-baseline)
      AUTO_BASELINE="false"; shift
      ;;
    --tmux-session)
      TMUX_SESSION="$2"; shift 2
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

# If --ckpt is provided, it overrides both jobs unless job-specific ckpts are also provided.
if [[ -n "$CKPT" ]]; then
  BEIJING_CKPT="$CKPT"
  PORTO_CKPT="$CKPT"
fi

if [[ -z "$BEIJING_CKPT" && -z "$PORTO_CKPT" ]]; then
  die "No checkpoints configured. Set BEIJING_CKPT/PORTO_CKPT in this script or pass --ckpt/--beijing-ckpt/--porto-ckpt."
fi

if [[ -n "$PRESET" ]]; then
  case "$PRESET" in
    per-type-detour)
      # Default: run both cities.
      if [[ -z "$JOBS" ]]; then
        JOBS="beijing,porto"
      fi
      ;;
    *)
      die "unknown --preset: $PRESET"
      ;;
  esac
fi

if [[ -z "$JOBS" ]]; then
  die "Provide --preset or --jobs"
fi

IFS="," read -r -a JOB_LIST <<<"$JOBS"
if [[ ${#JOB_LIST[@]} -eq 0 ]]; then
  die "--jobs parsed to an empty list"
fi

ensure_baseline() {
  local baseline_eval_dir="$1"
  local baseline_data_dir="$2"
  local baseline_dataset_name="$3"
  local lmtad_ckpt="$4"

  local baseline_json="$baseline_eval_dir/baseline_eval.json"
  if [[ -f "$baseline_json" ]]; then
    return 0
  fi
  if [[ "$AUTO_BASELINE" != "true" ]]; then
    die "missing baseline_eval.json at $baseline_json (auto-baseline disabled)"
  fi

  echo "[baseline] Creating baseline_eval.json for '$baseline_dataset_name' -> $baseline_eval_dir" >&2
  mkdir -p "$baseline_eval_dir"

  # This evaluates the baseline dataset split and writes <output-dir>/baseline_eval.json
  uv run python tools/evaluate_dataset_with_lmtad.py \
    --dataset "$baseline_dataset_name" \
    --data-dir "$baseline_data_dir" \
    --lmtad-checkpoint "$lmtad_ckpt" \
    --device "$DEVICE" \
    --batch-size "$BATCH_SIZE" \
    --splits "$BASELINE_SPLIT" \
    --output-dir "$baseline_eval_dir" \
    --write-baseline

  [[ -f "$baseline_json" ]] || die "baseline generation did not create: $baseline_json"
}

run_job() {
  local job="$1"

  local target_dir=""
  local baseline_eval=""
  local baseline_data_dir=""
  local baseline_dataset_name=""
  local job_ckpt=""

  case "$job" in
    beijing)
      target_dir="$BEIJING_TARGET_DIR"
      baseline_eval="$BEIJING_BASELINE_EVAL"
      baseline_data_dir="$BEIJING_BASELINE_DATA_DIR"
      baseline_dataset_name="Beijing"
      job_ckpt="$BEIJING_CKPT"
      ;;
    porto)
      target_dir="$PORTO_TARGET_DIR"
      baseline_eval="$PORTO_BASELINE_EVAL"
      baseline_data_dir="$PORTO_BASELINE_DATA_DIR"
      baseline_dataset_name="porto_hoser"
      job_ckpt="$PORTO_CKPT"
      ;;
    *)
      die "unknown job: $job (expected beijing or porto)"
      ;;
  esac

  [[ -d "$target_dir" ]] || die "target dir not found for job '$job': $target_dir"
  [[ -d "$baseline_data_dir" ]] || die "baseline data dir not found for job '$job': $baseline_data_dir"
  [[ -f "$job_ckpt" ]] || die "checkpoint not found for job '$job': $job_ckpt"

  ensure_baseline "$baseline_eval" "$baseline_data_dir" "$baseline_dataset_name" "$job_ckpt"

  mkdir -p "$OUT_DIR"
  local run_name="${NAME}_${job}"
  local log_path="$OUT_DIR/${run_name}.log"

  local -a cmd=(
    uv run python tools/run_lmtad_decision_benchmark.py
    --name "$run_name"
    --out-dir "$OUT_DIR"
    --baseline-eval "$baseline_eval"
    --baseline-data-dir "$baseline_data_dir"
    --baseline-split "$BASELINE_SPLIT"
    --split "$SPLIT"
    --device "$DEVICE"
    --batch-size "$BATCH_SIZE"
    --q "$Q_LIST"
    --ckpt "$job_ckpt"
    --target-data-dirs "$target_dir"
  )

  if [[ -n "$TMUX_SESSION" ]]; then
    return 0
  fi

  echo "Logging to: $log_path" >&2
  echo "Command: ${cmd[*]}" >&2
  "${cmd[@]}" 2>&1 | tee "$log_path"
}

run_job_in_tmux() {
  local job="$1"

  local target_dir=""
  local baseline_eval=""
  local baseline_data_dir=""
  local baseline_dataset_name=""
  local job_ckpt=""

  case "$job" in
    beijing)
      target_dir="$BEIJING_TARGET_DIR"
      baseline_eval="$BEIJING_BASELINE_EVAL"
      baseline_data_dir="$BEIJING_BASELINE_DATA_DIR"
      baseline_dataset_name="Beijing"
      job_ckpt="$BEIJING_CKPT"
      ;;
    porto)
      target_dir="$PORTO_TARGET_DIR"
      baseline_eval="$PORTO_BASELINE_EVAL"
      baseline_data_dir="$PORTO_BASELINE_DATA_DIR"
      baseline_dataset_name="porto_hoser"
      job_ckpt="$PORTO_CKPT"
      ;;
    *)
      die "unknown job: $job (expected beijing or porto)"
      ;;
  esac

  [[ -d "$target_dir" ]] || die "target dir not found for job '$job': $target_dir"
  [[ -d "$baseline_data_dir" ]] || die "baseline data dir not found for job '$job': $baseline_data_dir"
  [[ -f "$job_ckpt" ]] || die "checkpoint not found for job '$job': $job_ckpt"

  ensure_baseline "$baseline_eval" "$baseline_data_dir" "$baseline_dataset_name" "$job_ckpt"

  mkdir -p "$OUT_DIR"
  local run_name="${NAME}_${job}"
  local log_path="$OUT_DIR/${run_name}.log"

  # Run inside tmux window via a shell command (needed for tee/piping).
  local shell_cmd="cd \"$ROOT_DIR\" && uv run python tools/run_lmtad_decision_benchmark.py \
  --name \"$run_name\" \
  --out-dir \"$OUT_DIR\" \
  --baseline-eval \"$baseline_eval\" \
  --baseline-data-dir \"$baseline_data_dir\" \
  --baseline-split \"$BASELINE_SPLIT\" \
  --split \"$SPLIT\" \
  --device \"$DEVICE\" \
  --batch-size \"$BATCH_SIZE\" \
  --q \"$Q_LIST\" \
  --ckpt \"$job_ckpt\" \
  --target-data-dirs \"$target_dir\" \
  2>&1 | tee \"$log_path\""

  tmux new-window -t "$TMUX_SESSION" -n "$run_name" "$shell_cmd"
}

if [[ -n "$TMUX_SESSION" ]]; then
  command -v tmux >/dev/null 2>&1 || die "tmux not found in PATH"
  # Create the session if it does not exist; otherwise reuse it.
  if ! tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    tmux new-session -d -s "$TMUX_SESSION" -n status "cd \"$ROOT_DIR\" && echo 'Session: $TMUX_SESSION' && echo 'Attach with: tmux attach -t $TMUX_SESSION' && bash"
  fi

  for job in "${JOB_LIST[@]}"; do
    job="$(echo "$job" | xargs)"
    [[ -n "$job" ]] || continue
    run_job_in_tmux "$job"
  done

  echo "Started tmux session: $TMUX_SESSION" >&2
  echo "Attach: tmux attach -t $TMUX_SESSION" >&2
  echo "Detach: Ctrl-b d" >&2
  exit 0
fi

for job in "${JOB_LIST[@]}"; do
  job="$(echo "$job" | xargs)"
  [[ -n "$job" ]] || continue
  run_job "$job"
done
