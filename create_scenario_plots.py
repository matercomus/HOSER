#!/usr/bin/env python3
"""
Generate all scenario analysis plots.

Usage:
    uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6
"""

import argparse
import logging
from pathlib import Path

from scenario_plots.data_loader import ScenarioDataLoader
from scenario_plots import metrics_plots
from scenario_plots import temporal_spatial_plots
from scenario_plots import robustness_plots
from scenario_plots import analysis_plots
from scenario_plots import application_plots

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Generate scenario analysis plots',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate plots for Beijing evaluation
  uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-6
  
  # Generate plots for Porto evaluation
  uv run python create_scenario_plots.py --eval-dir hoser-distill-optuna-porto-eval-xxx
        """
    )
    parser.add_argument('--eval-dir', required=True, help='Evaluation directory containing scenarios/')
    parser.add_argument('--dpi', type=int, default=300, help='Output resolution (default: 300)')
    
    args = parser.parse_args()
    
    eval_dir = Path(args.eval_dir).resolve()
    output_dir = eval_dir / 'figures' / 'scenarios'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    loader = ScenarioDataLoader(eval_dir)
    data = loader.load_all()
    
    logger.info("\n📊 Generating scenario visualizations...")
    
    # Metrics plots (4 plots: #1, #2, #3, #8)
    metrics_plots.plot_all(data, output_dir, args.dpi)
    
    # Temporal/Spatial plots (2 plots: #4, #5)
    temporal_spatial_plots.plot_all(data, output_dir, args.dpi)
    
    # Robustness plots (2 plots: #6, #7)
    robustness_plots.plot_all(data, output_dir, args.dpi)
    
    # Analysis plots (3 plots: #9, #10, #11)
    analysis_plots.plot_all(data, output_dir, args.dpi)
    
    # Application plots (2 plots: #12, #13)
    application_plots.plot_all(data, output_dir, args.dpi)
    
    logger.info(f"\n✅ All 13 plots saved to {output_dir}")


if __name__ == '__main__':
    main()

