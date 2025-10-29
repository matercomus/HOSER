#!/usr/bin/env python3
"""
Generate and save all Optuna visualization plots for the documentation.

This script loads the Optuna study and generates publication-quality plots
for the hyperparameter optimization documentation.

Usage:
    # Beijing study (default)
    uv run python tools/generate_optuna_plots.py

    # Porto study using preset
    uv run python tools/generate_optuna_plots.py --preset porto

    # Custom study
    uv run python tools/generate_optuna_plots.py --study your_study_name --output your/output/dir
"""

import argparse
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour,
    plot_edf,
    plot_timeline,
)
import os
import plotly.io as pio

# Try to use kaleido for static export
try:
    pio.kaleido.scope.chromium_args = ["--no-sandbox", "--disable-gpu"]
except Exception:
    pass

# Default configuration
DEFAULT_STUDY_NAME = "hoser_tuning_20251003_162916"  # Beijing study
DEFAULT_STORAGE_URL = (
    "sqlite:////mnt/i/Matt-Backups/HOSER-Backups/HOSER-Distil/optuna_hoser.db"
)
DEFAULT_OUTPUT_DIR = "docs/figures/optuna"

# Study presets
STUDY_PRESETS = {
    "beijing": {
        "study_name": "hoser_tuning_20251003_162916",
        "output_dir": "docs/figures/optuna",
        "description": "Beijing Taxi Dataset (Oct 3-6, 2025)",
    },
    "porto": {
        "study_name": "hoser_tuning_20251014_145134",
        "output_dir": "hoser-distill-optuna-porto-eval-eb0e88ab-20251026_152732/figures/optuna",
        "description": "Porto Taxi Dataset (Oct 14-19, 2025)",
    },
}


def main():
    """Generate and save all Optuna plots."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate Optuna visualization plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Beijing study (default)
  uv run python tools/generate_optuna_plots.py
  
  # Porto study using preset
  uv run python tools/generate_optuna_plots.py --preset porto
  
  # Custom study
  uv run python tools/generate_optuna_plots.py --study your_study_name --output your/output/dir
        """,
    )

    parser.add_argument(
        "--preset",
        choices=["beijing", "porto"],
        help="Use predefined study configuration",
    )
    parser.add_argument("--study", help="Optuna study name (overrides preset)")
    parser.add_argument(
        "--storage",
        default=DEFAULT_STORAGE_URL,
        help="Optuna storage URL (default: %(default)s)",
    )
    parser.add_argument(
        "--output", help="Output directory for plots (overrides preset)"
    )

    args = parser.parse_args()

    # Determine configuration from preset or arguments
    if args.preset:
        preset = STUDY_PRESETS[args.preset]
        study_name = args.study or preset["study_name"]
        output_dir = args.output or preset["output_dir"]
        description = preset["description"]
        print(f"📋 Using preset: {args.preset} - {description}")
    else:
        study_name = args.study or DEFAULT_STUDY_NAME
        output_dir = args.output or DEFAULT_OUTPUT_DIR

    storage_url = args.storage

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")

    # Load study
    print(f"📊 Loading study: {study_name}")
    study = optuna.load_study(study_name=study_name, storage=storage_url)

    print(f"✅ Loaded {len(study.trials)} trials")
    print(f"🏆 Best trial: {study.best_trial.number}")
    print(f"📈 Best value: {study.best_value:.6f}")
    print(f"🔧 Best params: {study.best_params}")
    print()

    # Generate plots
    plots = [
        ("optimization_history", plot_optimization_history, "Optimization History"),
        ("param_importance", plot_param_importances, "Parameter Importance"),
        ("parallel_coordinate", plot_parallel_coordinate, "Parallel Coordinates"),
        ("slice_plot", plot_slice, "Slice Plot"),
        ("contour_plot", plot_contour, "Contour Plot"),
        ("edf_plot", plot_edf, "EDF Plot"),
        ("timeline", plot_timeline, "Timeline"),
    ]

    for filename, plot_func, title in plots:
        print(f"🎨 Generating {title}...")
        try:
            fig = plot_func(study)
            # Save as HTML (interactive, no Chrome required)
            html_path = os.path.join(output_dir, f"{filename}.html")
            fig.write_html(html_path)
            print(f"   ✅ Saved HTML to {html_path}")

            # Also try to save as PNG if kaleido works
            try:
                png_path = os.path.join(output_dir, f"{filename}.png")
                fig.write_image(png_path, width=1000, height=600)
                print(f"   ✅ Saved PNG to {png_path}")
            except Exception:
                print("   ⚠️  PNG export failed (kaleido/Chrome not available)")
                print(
                    f"      To generate PNG: open {html_path} in browser and screenshot"
                )
        except Exception as e:
            print(f"   ❌ Error: {e}")

    print()
    print("🎉 All plots generated successfully!")
    print(f"📂 View plots in: {output_dir}")
    print()
    print("📝 Next steps:")
    print("   1. Open HTML files in browser to view interactive plots")
    print("   2. Take screenshots for markdown documentation")
    print("   3. Update relevant Hyperparameter-Optimization*.md with plot references")


if __name__ == "__main__":
    main()
