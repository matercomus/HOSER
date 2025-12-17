"""Fix the repository's local `data/` layout.

This repo expects datasets to live at `data/<dataset_name>/`.

On some systems the repo ends up with a nested symlink `data/data -> /path/to/real/data`.
That layout is awkward because code and docs consistently refer to `data/<dataset>/...`.

This script removes the nested symlink and replaces it with per-dataset symlinks:

- `data/Beijing -> /path/to/real/data/Beijing`
- `data/porto_hoser -> /path/to/real/data/porto_hoser`
- ...

It is safe by default:
- It will not overwrite existing non-symlink paths.
- It will not remove anything except the nested `data/data` symlink.

Example:
    uv run python scripts/fix_data_symlinks.py

    # Or explicitly specify where the real datasets live
    uv run python scripts/fix_data_symlinks.py --real-data-dir /local/data/mka299/hoser/data

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class FixResult:
    """Summary of actions taken."""

    removed_nested_symlink: bool
    created: tuple[str, ...]
    skipped: tuple[str, ...]


def _repo_root() -> Path:
    """Return the repository root based on this script's location."""

    return Path(__file__).resolve().parents[1]


def _iter_dataset_dirs(real_data_dir: Path) -> Iterable[Path]:
    """Yield direct child directories of the real data dir."""

    for child in sorted(real_data_dir.iterdir()):
        if child.name.startswith("."):
            continue
        if child.is_dir():
            yield child


def _unlink_if_symlink(path: Path) -> bool:
    """Unlink *only* if `path` is a symlink.

    Returns:
        True if a symlink was removed, else False.
    """

    if path.is_symlink():
        path.unlink()
        return True
    return False


def fix_data_symlinks(
    *,
    dest_dir: Path,
    real_data_dir: Path | None,
    force: bool,
    dry_run: bool,
) -> FixResult:
    """Fix `dest_dir` to contain per-dataset symlinks.

    Args:
        dest_dir: Destination directory, usually `<repo>/data`.
        real_data_dir: Directory containing the real datasets.
            If omitted, and `<dest_dir>/data` is a symlink, it will be used.
        force: If True, overwrite existing symlinks at `data/<dataset>`.
        dry_run: If True, print what would be done without making changes.
    """

    nested_link = dest_dir / "data"

    resolved_real_data_dir: Path | None = real_data_dir
    if resolved_real_data_dir is None and nested_link.is_symlink():
        resolved_real_data_dir = nested_link.resolve()

    if resolved_real_data_dir is None:
        raise ValueError(
            "Could not determine real data dir. Provide --real-data-dir, "
            "or create a symlink at data/data first."
        )

    if not resolved_real_data_dir.exists():
        raise FileNotFoundError(
            f"Real data dir does not exist: {resolved_real_data_dir}"
        )

    removed_nested_symlink = False
    if nested_link.exists() or nested_link.is_symlink():
        if not nested_link.is_symlink():
            raise ValueError(
                f"Refusing to remove non-symlink path: {nested_link}"
            )
        if dry_run:
            print(f"[dry-run] Would remove nested symlink: {nested_link}")
        else:
            removed_nested_symlink = _unlink_if_symlink(nested_link)

    created: list[str] = []
    skipped: list[str] = []

    for dataset_dir in _iter_dataset_dirs(resolved_real_data_dir):
        link_path = dest_dir / dataset_dir.name

        if link_path.exists() or link_path.is_symlink():
            if link_path.is_symlink() and force:
                if dry_run:
                    print(f"[dry-run] Would overwrite symlink: {link_path}")
                else:
                    link_path.unlink()
            else:
                skipped.append(dataset_dir.name)
                continue

        if dry_run:
            print(f"[dry-run] Would link {link_path} -> {dataset_dir}")
        else:
            link_path.symlink_to(dataset_dir)
        created.append(dataset_dir.name)

    return FixResult(
        removed_nested_symlink=removed_nested_symlink,
        created=tuple(created),
        skipped=tuple(skipped),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replace a nested `data/data` symlink with per-dataset symlinks "
            "in `data/`."
        )
    )
    parser.add_argument(
        "--real-data-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the real datasets (e.g. /local/data/.../hoser/data). "
            "If omitted, uses the target of `data/data` if it exists as a symlink."
        ),
    )
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=None,
        help="Destination directory (default: <repo>/data).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing symlinks for datasets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without making changes.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    root = _repo_root()
    dest_dir = args.dest_dir if args.dest_dir is not None else root / "data"

    result = fix_data_symlinks(
        dest_dir=dest_dir,
        real_data_dir=args.real_data_dir,
        force=args.force,
        dry_run=args.dry_run,
    )

    print(
        "✅ Done. "
        f"Removed nested symlink: {result.removed_nested_symlink}. "
        f"Created: {len(result.created)}. "
        f"Skipped: {len(result.skipped)}."
    )
    if result.skipped:
        print(f"Skipped datasets: {', '.join(result.skipped)}")


if __name__ == "__main__":
    main()
