#!/usr/bin/env python3
"""
Generate and save all Optuna visualization plots for the documentation.

This script loads the Optuna study and generates publication-quality plots
for the hyperparameter optimization documentation.

Usage:
    uv run python tools/generate_optuna_plots.py
"""

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
    import kaleido
    pio.kaleido.scope.chromium_args = ["--no-sandbox", "--disable-gpu"]
except Exception:
    pass

# Configuration
STUDY_NAME = "hoser_tuning_20251003_162916"
STORAGE_URL = "sqlite:////mnt/i/Matt-Backups/HOSER-Backups/HOSER-Distil/optuna_hoser.db"
OUTPUT_DIR = "docs/figures/optuna"

def main():
    """Generate and save all Optuna plots."""
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    # Load study
    print(f"📊 Loading study: {STUDY_NAME}")
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL
    )
    
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
            html_path = os.path.join(OUTPUT_DIR, f"{filename}.html")
            fig.write_html(html_path)
            print(f"   ✅ Saved HTML to {html_path}")
            
            # Also try to save as PNG if kaleido works
            try:
                png_path = os.path.join(OUTPUT_DIR, f"{filename}.png")
                fig.write_image(png_path, width=1000, height=600)
                print(f"   ✅ Saved PNG to {png_path}")
            except Exception:
                print(f"   ⚠️  PNG export failed (kaleido/Chrome not available)")
                print(f"      To generate PNG: open {html_path} in browser and screenshot")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print()
    print("🎉 All plots generated successfully!")
    print(f"📂 View plots in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

