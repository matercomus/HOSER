#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  link_best_checkpoints.sh --save-dir <dir> --models-dir <dir> [--dry-run] [--force]

Description:
  For each immediate child directory under --save-dir, this script looks for
  a checkpoint file named 'best.pth' and creates a symlink in --models-dir
  named after the child directory:

    <save-dir>/<run-name>/best.pth -> <models-dir>/<run-name>.pth

  Notes:
  - --save-dir may itself be a symlink; the link target is resolved with
    readlink -f so symlinks always point at the real checkpoint file.
  - By default, the script refuses to overwrite an existing destination file.

Options:
  --save-dir    Directory containing run subfolders (depth=1)
  --models-dir  Destination models directory (will be created if missing)
  --dry-run     Print intended actions; do not modify the filesystem
  --force       Overwrite existing destination paths
  -h, --help    Show this help

Examples:
  # Preview actions
  scripts/link_best_checkpoints.sh \
    --save-dir save/Beijing_abnormal_3 \
    --models-dir hoser-perturbed-beijing-abnormal-3-eval/models \
    --dry-run

  # Perform symlinks
  scripts/link_best_checkpoints.sh \
    --save-dir save/Beijing_abnormal_3 \
    --models-dir hoser-perturbed-beijing-abnormal-3-eval/models \
    --force
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

save_dir=""
models_dir=""
dry_run=0
force=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --save-dir)
      save_dir="${2:-}"; shift 2 ;;
    --models-dir)
      models_dir="${2:-}"; shift 2 ;;
    --dry-run)
      dry_run=1; shift ;;
    --force)
      force=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      die "Unknown argument: $1 (use --help)" ;;
  esac
done

[[ -n "$save_dir" ]] || die "--save-dir is required"
[[ -n "$models_dir" ]] || die "--models-dir is required"

save_dir_real="$(readlink -f "$save_dir")" || die "Failed to resolve --save-dir: $save_dir"
models_dir_real="$(readlink -f "$models_dir" 2>/dev/null || true)"

if [[ ! -d "$save_dir_real" ]]; then
  die "--save-dir is not a directory: $save_dir_real"
fi

if [[ -z "$models_dir_real" ]]; then
  models_dir_real="$models_dir"
fi

if [[ $dry_run -eq 1 ]]; then
  echo "[dry-run] mkdir -p '$models_dir_real'"
else
  mkdir -p "$models_dir_real"
fi

shopt -s nullglob
found_any=0
linked_any=0

for run_dir in "$save_dir_real"/*/; do
  [[ -d "$run_dir" ]] || continue
  run_dir_no_slash="${run_dir%/}"
  run_name="$(basename "$run_dir_no_slash")"
  src="$run_dir_no_slash/best.pth"
  if [[ ! -f "$src" ]]; then
    echo "Skipping (missing best.pth): $run_dir" >&2
    continue
  fi

  found_any=1
  dest="$models_dir_real/$run_name.pth"

  if [[ -e "$dest" || -L "$dest" ]]; then
    if [[ $force -ne 1 ]]; then
      echo "Refusing to overwrite existing: $dest (use --force)" >&2
      continue
    fi
  fi

  if [[ $dry_run -eq 1 ]]; then
    echo "[dry-run] ln -sfn '$src' '$dest'"
  else
    ln -sfn "$src" "$dest"
  fi

  linked_any=1
done

if [[ $found_any -eq 0 ]]; then
  die "No run subdirectories with best.pth found under: $save_dir_real"
fi

if [[ $linked_any -eq 0 ]]; then
  echo "No links created (everything skipped)." >&2
fi
