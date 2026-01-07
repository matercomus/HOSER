#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Prepare per-abnormality-type datasets (detour-only / route_switch-only / perturb-only).

This wraps tools/prepare_per_type_abnormal_datasets.py and applies the same
UV_PROJECT_ENVIRONMENT bootstrapping used by scripts/train_perturbed_experiment.sh.

Required:
  --input-dir DIR    Base dataset dir (e.g. data/Beijing)
  --out-root DIR     Output root (e.g. data/_per_type)

Optional:
  --types LIST       Comma-separated types (default: detour,route_switch,perturb)
  --splits LIST      Comma-separated splits (default: train,val,test)
  --rate R           Abnormality rate per input row (default: 0.05)
  --level LEVEL      low|medium|high (default: medium)
  --seed N           RNG seed (default: 42)
  --ensure-change    Retry types until a change is made (recommended)
  --strong-prob P    Probability of 'strong' anomaly variant (default: 0.3)

Example:
  bash scripts/prepare_per_type_abnormal_datasets.sh \
    --input-dir data/Beijing \
    --out-root data/_per_type \
    --rate 0.05 \
    --ensure-change
EOF
}

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

INPUT_DIR=""
OUT_ROOT=""
TYPES="detour"
SPLITS="train"
RATE="0.15"
LEVEL="strong"
SEED="42"
ENSURE_CHANGE="1"
STRONG_PROB="0.5"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --input-dir)
      INPUT_DIR="$2"; shift 2
      ;;
    --out-root)
      OUT_ROOT="$2"; shift 2
      ;;
    --types)
      TYPES="$2"; shift 2
      ;;
    --splits)
      SPLITS="$2"; shift 2
      ;;
    --rate)
      RATE="$2"; shift 2
      ;;
    --level)
      LEVEL="$2"; shift 2
      ;;
    --seed)
      SEED="$2"; shift 2
      ;;
    --ensure-change)
      ENSURE_CHANGE="1"; shift
      ;;
    --strong-prob)
      STRONG_PROB="$2"; shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$INPUT_DIR" || -z "$OUT_ROOT" ]]; then
  echo "Missing required --input-dir and/or --out-root" >&2
  usage >&2
  exit 2
fi

CMD=(uv run python tools/prepare_per_type_abnormal_datasets.py
  --input-dir "$INPUT_DIR"
  --out-root "$OUT_ROOT"
  --types "$TYPES"
  --splits "$SPLITS"
  --abnormality-rate "$RATE"
  --level "$LEVEL"
  --seed "$SEED"
  --strong-prob "$STRONG_PROB"
)

if [[ "$ENSURE_CHANGE" == "1" ]]; then
  CMD+=(--ensure-change)
fi

echo "Command: ${CMD[*]}" >&2
"${CMD[@]}"
