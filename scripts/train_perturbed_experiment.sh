#!/usr/bin/env bash
set -euo pipefail

# Train the perturbed-data experiment models for both datasets.
#
# Defaults are chosen to match the thesis methodology note:
# - Additive loss in train_with_distill.py: CE + time + lambda * KL
# - vanilla: no distill
# - distilled_l0p001: weak distill (lambda ~ 0.001)
# - distilled_l0p5: medium distill (lambda ~ 0.5)
# - distilled_l1: strong distill (lambda ~ 1.0 by default for additive formulation)
#
# Usage:
#   bash scripts/train_perturbed_experiment.sh
#
# Optional env vars:
#   CUDA=0
#   SEEDS="42 43 44"
#   WEAK_LAMBDA=0.001
#   MEDIUM_LAMBDA=0.5
#   STRONG_LAMBDA=1.0
#   DATASET=beijing|porto|all   # filter which dataset(s) to run (default: all)
#   LOG_ROOT=logs/perturbed_training  # where per-run logs are written
#   DRY_RUN=1            # print commands only
#   BACKUP_EXISTING=1    # move existing save/<dataset>/seed*_{vanilla,distill*} aside

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_perturbed_experiment.sh [all|beijing|porto]

If no argument is provided, uses $DATASET (env) or defaults to 'all'.

Examples:
  DRY_RUN=1 bash scripts/train_perturbed_experiment.sh beijing
  CUDA=0 SEEDS="42" bash scripts/train_perturbed_experiment.sh porto
  CUDA=0 SEEDS="42 43 44" bash scripts/train_perturbed_experiment.sh all
EOF
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Non-interactive scripts do not source ~/.bashrc, so they won't pick up the
# user's `uv()` wrapper. Ensure uv uses the per-project venv under /local.
if [[ -z "${UV_PROJECT_ENVIRONMENT:-}" ]]; then
  root_real="$(readlink -f "$ROOT_DIR")"
  hash="$(printf '%s' "$root_real" | sha1sum | awk '{print substr($1,1,8)}')"
  name="$(basename "$root_real")-$hash"
  envdir="/local/data/mka299/uv/venvs/$name"
  mkdir -p "$envdir" 2>/dev/null || true
  export UV_PROJECT_ENVIRONMENT="$envdir"
fi

CUDA="${CUDA:-0}"
SEEDS="${SEEDS:-42}"
WEAK_LAMBDA="${WEAK_LAMBDA:-0.001}"
MEDIUM_LAMBDA="${MEDIUM_LAMBDA:-0.5}"
STRONG_LAMBDA="${STRONG_LAMBDA:-1.0}"
DRY_RUN="${DRY_RUN:-0}"
BACKUP_EXISTING="${BACKUP_EXISTING:-0}"
LOG_ROOT="${LOG_ROOT:-logs/perturbed_training}"

DATASET_SELECTOR_ARG="${1:-}"
if [[ "$DATASET_SELECTOR_ARG" == "-h" || "$DATASET_SELECTOR_ARG" == "--help" ]]; then
  usage
  exit 0
fi

DATASET_SELECTOR="${DATASET_SELECTOR_ARG:-${DATASET:-all}}"
case "$DATASET_SELECTOR" in
  all|beijing|porto) ;;
  *)
    echo "ERROR: unknown dataset selector: '$DATASET_SELECTOR'" >&2
    usage
    exit 2
    ;;
esac

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY_RUN] %q ' "$@"; echo
  else
    "$@"
  fi
}

run_with_tee() {
  # Run a command, streaming output to console and appending to a log file.
  # Returns the underlying command's exit code.
  local log_file="$1"; shift

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN] log: $log_file"
    printf '[DRY_RUN] %q ' "$@"; echo
    return 0
  fi

  mkdir -p "$(dirname "$log_file")"
  echo "Logging to: $log_file"

  # Ensure timely log flushing from Python.
  # With pipefail, a pipeline can mask the producer's status; use PIPESTATUS.
  set +e
  PYTHONUNBUFFERED=1 "$@" 2>&1 | tee -a "$log_file"
  local rc=${PIPESTATUS[0]}
  set -e
  return $rc
}

require_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing required file: $path" >&2
    exit 1
  fi
}

maybe_backup_saves() {
  local dataset_name="$1"
  if [[ "$BACKUP_EXISTING" != "1" ]]; then
    return 0
  fi

  local save_root="save/${dataset_name}"
  if [[ ! -d "$save_root" ]]; then
    return 0
  fi

  # Move any seed directories for this dataset out of the way
  local timestamp
  timestamp="$(date +%Y%m%d_%H%M%S)"
  local backup_dir="${save_root}/_backup_${timestamp}"

  shopt -s nullglob
  # train_with_distill.py may include lambda tokens in the directory suffix
  # (e.g., seed42_distill_l0p001), so back up distill*.
  local dirs=("${save_root}"/seed*_vanilla* "${save_root}"/seed*_distill*)
  shopt -u nullglob

  if [[ ${#dirs[@]} -eq 0 ]]; then
    return 0
  fi

  echo "Backing up existing checkpoints for ${dataset_name} -> ${backup_dir}" >&2
  run_cmd mkdir -p "$backup_dir"
  for d in "${dirs[@]}"; do
    run_cmd mv "$d" "$backup_dir/"
  done
}

check_dataset_dir() {
  local dataset_dir="$1"
  require_file "$dataset_dir/roadmap.geo"
  require_file "$dataset_dir/roadmap.rel"
  require_file "$dataset_dir/road_network_partition"
  require_file "$dataset_dir/zone_trans_mat.npy"
  require_file "$dataset_dir/train.csv"
  require_file "$dataset_dir/val.csv"
  require_file "$dataset_dir/test.csv"
}

train_one() {
  local dataset_name="$1"   # used for save/<dataset_name>/...
  local data_dir="$2"       # points at data/<dataset>
  local config_path="$3"    # config/<base>.yaml
  local seed="$4"
  local variant="$5"        # vanilla|distilled_l0p001|distilled_l0p5|distilled_l1

  local -a cmd=(uv run python train_with_distill.py
    --dataset "$dataset_name"
    --config "$config_path"
    --seed "$seed"
    --cuda "$CUDA"
    --data_dir "$data_dir")

  if [[ "$variant" == "vanilla" ]]; then
    cmd+=(--no-distill)
  elif [[ "$variant" == "distilled_l0p001" ]]; then
    cmd+=(--distill-lambda "$WEAK_LAMBDA")
  elif [[ "$variant" == "distilled_l0p5" ]]; then
    cmd+=(--distill-lambda "$MEDIUM_LAMBDA")
  elif [[ "$variant" == "distilled_l1" ]]; then
    cmd+=(--distill-lambda "$STRONG_LAMBDA")
  else
    echo "ERROR: unknown variant: $variant" >&2
    exit 1
  fi

  echo "---"
  echo "Training ${dataset_name} seed=${seed} ${variant} (data_dir=${data_dir})"

  local log_dir="${LOG_ROOT}/${dataset_name}/seed${seed}"
  local log_file="${log_dir}/${variant}.log"
  local start_ts end_ts
  start_ts="$(date +%Y-%m-%dT%H:%M:%S)"
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$log_dir"
    echo "Start: ${start_ts}" | tee -a "$log_file" >/dev/null || true
  fi

  if run_with_tee "$log_file" "${cmd[@]}"; then
    :
  else
    local rc=$?
    echo "ERROR: training failed (${dataset_name} seed=${seed} ${variant}) rc=${rc}" >&2
    exit $rc
  fi

  end_ts="$(date +%Y-%m-%dT%H:%M:%S)"
  if [[ "$DRY_RUN" != "1" ]]; then
    echo "End: ${end_ts}" | tee -a "$log_file" >/dev/null || true
  fi
}

# Dataset table: (dataset_name, data_dir, config_path)
DATASETS=()
if [[ "$DATASET_SELECTOR" == "all" || "$DATASET_SELECTOR" == "beijing" ]]; then
  DATASETS+=("Beijing_abnormal_3|data/Beijing_abnormal_3|config/Beijing.yaml")
fi
if [[ "$DATASET_SELECTOR" == "all" || "$DATASET_SELECTOR" == "porto" ]]; then
  DATASETS+=("porto_hoser_abnormal_3|data/porto_hoser_abnormal_3|config/porto_hoser.yaml")
fi

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "ERROR: no datasets selected (selector='$DATASET_SELECTOR')" >&2
  exit 2
fi

# Preflight
for entry in "${DATASETS[@]}"; do
  IFS='|' read -r dataset_name data_dir config_path <<<"$entry"
  require_file "$config_path"
  check_dataset_dir "$data_dir"
  maybe_backup_saves "$dataset_name"
  echo "OK: ${dataset_name} (${data_dir})"
  echo "  config: ${config_path}"
  echo "  seeds:  ${SEEDS}"
  echo "  CUDA:   ${CUDA}"
  echo "  lambdas: weak=${WEAK_LAMBDA} medium=${MEDIUM_LAMBDA} strong=${STRONG_LAMBDA}"
  echo

done
# Train
# Seed-major ordering: finish one seed across all datasets (M1/M2/M3) before
# moving to the next seed so you can start validating the first seed's outputs.
for seed in $SEEDS; do
  for entry in "${DATASETS[@]}"; do
    IFS='|' read -r dataset_name data_dir config_path <<<"$entry"
    train_one "$dataset_name" "$data_dir" "$config_path" "$seed" vanilla
    train_one "$dataset_name" "$data_dir" "$config_path" "$seed" distilled_l0p001
    train_one "$dataset_name" "$data_dir" "$config_path" "$seed" distilled_l0p5
    train_one "$dataset_name" "$data_dir" "$config_path" "$seed" distilled_l1
  done
done

echo "---"
echo "All training runs completed."
