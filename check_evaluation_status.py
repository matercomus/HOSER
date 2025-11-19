#!/usr/bin/env python3
"""
Check the status of the LM-TAD evaluation process
"""

import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("/home/matt/Dev/HOSER")
EVAL_DIR = BASE_DIR / "hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732" / "eval_lmtad_simple" / "porto_hoser"

def check_status():
    print("\n" + "=" * 80)
    print("LM-TAD Evaluation Status Check")
    print("=" * 80 + "\n")

    # Check parent directories
    print("Directory Status:")
    if EVAL_DIR.exists():
        print(f"  ✓ Evaluation directory exists: {EVAL_DIR}")
    else:
        print(f"  ✗ Evaluation directory not found: {EVAL_DIR}")
        return

    # Check for results file
    results_file = EVAL_DIR / "evaluation_results.json"
    print("\nResults File Status:")
    if results_file.exists():
        stat = results_file.stat()
        mtime = datetime.fromtimestamp(stat.st_mtime)
        age = datetime.now() - mtime
        print(f"  ✓ Found: {results_file}")
        print(f"    - Size: {stat.st_size:,} bytes")
        print(f"    - Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')} ({age} ago)")

        # Try to load and show structure
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            print(f"    - Contains {len(data)} top-level keys")
            print(f"    - Keys: {', '.join(list(data.keys())[:5])}{'...' if len(data) > 5 else ''}")
        except Exception as e:
            print(f"    ⚠ Could not parse JSON: {e}")
    else:
        print(f"  ✗ Not found: {results_file}")

    # Check for CSV files
    csv_files = list(EVAL_DIR.glob("*.csv"))
    print("\nCSV Files Status:")
    if csv_files:
        print(f"  ✓ Found {len(csv_files)} CSV files:")
        for csv_file in sorted(csv_files):
            stat = csv_file.stat()
            size_kb = stat.st_size / 1024
            print(f"    - {csv_file.name} ({size_kb:.1f} KB)")
    else:
        print("  ✗ No CSV files found")

    # Check figures directory
    figures_dir = EVAL_DIR / "figures"
    print("\nFigures Directory Status:")
    if figures_dir.exists():
        fig_files = list(figures_dir.glob("*.png"))
        print(f"  ✓ Figures directory exists with {len(fig_files)} PNG files")
        for fig_file in sorted(fig_files):
            print(f"    - {fig_file.name}")
    else:
        print("  ✗ Figures directory does not exist yet")

    # Check for log files in parent
    log_dir = BASE_DIR / "hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732"
    log_files = list(log_dir.glob("*.log")) + list(log_dir.glob("*.out"))
    print("\nLog Files in Parent Directory:")
    if log_files:
        for log_file in sorted(log_files):
            stat = log_file.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime)
            size_kb = stat.st_size / 1024
            print(f"  - {log_file.name} ({size_kb:.1f} KB, modified {mtime})")
    else:
        print("  - No log files found")

    print("\n" + "-" * 80)
    print("NEXT STEPS")
    print("-" * 80 + "\n")

    if results_file.exists():
        print("✓ Results are available! Run the visualization script:")
        print("  python /home/matt/Dev/HOSER/create_lmtad_visualizations.py\n")
    else:
        print("⏳ Evaluation still running. Options:")
        print("  1. Monitor with the script:")
        print("     bash /home/matt/Dev/HOSER/monitor_results.sh")
        print("  2. Check this status again later")
        print("  3. Check the training logs for progress\n")

    print("=" * 80 + "\n")


if __name__ == "__main__":
    check_status()