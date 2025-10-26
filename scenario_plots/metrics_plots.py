"""
Metric comparison plots for scenario analysis.

Plots:
- #1: Scenario Metrics Heatmap (6-panel)
- #2: Train OD Scenario Comparison (grouped bar)
- #3: Test OD Scenario Comparison (grouped bar)
- #8: Metric Sensitivity Grid (3×3)
"""

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .data_loader import get_metric_value, get_scenario_list

logger = logging.getLogger(__name__)

# Style configuration
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

COLORS = {
    'vanilla': '#e74c3c',
    'distilled': '#3498db',
    'distilled_seed44': '#2ecc71'
}

MODEL_LABELS = {
    'vanilla': 'Vanilla',
    'distilled': 'Distilled (seed 42)',
    'distilled_seed44': 'Distilled (seed 44)'
}


def plot_all(data: Dict, output_dir: Path, dpi: int = 300):
    """Generate all metrics plots"""
    logger.info("  📈 Metrics plots...")
    
    plot_scenario_metrics_heatmap(data, output_dir, dpi)
    plot_od_scenario_comparison(data, 'train', output_dir, dpi)
    plot_od_scenario_comparison(data, 'test', output_dir, dpi)
    plot_metric_sensitivity(data, output_dir, dpi)


def plot_scenario_metrics_heatmap(data: Dict, output_dir: Path, dpi: int):
    """Plot #1: 6-panel heatmap of all metrics × scenarios × models"""
    logger.info("    1. Scenario metrics heatmap")
    
    metrics = ['Distance_JSD', 'Duration_JSD', 'Radius_JSD', 
               'Hausdorff_km', 'DTW_km', 'EDR']
    metric_labels = ['Distance JSD', 'Duration JSD', 'Radius of Gyration JSD',
                     'Hausdorff Distance (km)', 'DTW Distance (km)', 'EDR']
    
    # Get scenarios and models
    scenarios = get_scenario_list(data, 'train', 'vanilla')
    scenarios = [s for s in scenarios if s in ['off_peak', 'peak', 'weekday', 
                                                'weekend', 'city_center', 'suburban']]
    models = ['vanilla', 'distilled', 'distilled_seed44']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Scenario Performance Metrics: All Models', fontsize=16, fontweight='bold', y=0.995)
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 3, idx % 3]
        
        # Build matrix: scenarios × models
        matrix = np.zeros((len(scenarios), len(models)))
        for i, scen in enumerate(scenarios):
            for j, model in enumerate(models):
                val = get_metric_value(data, 'train', model, scen, metric)
                matrix[i, j] = val if val is not None else np.nan
        
        # Normalize for coloring (lower is better)
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn_r',
                   xticklabels=[MODEL_LABELS[m] for m in models],
                   yticklabels=[s.replace('_', ' ').title() for s in scenarios],
                   ax=ax, cbar_kws={'label': 'Value'}, vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_path = output_dir / 'scenario_metrics_heatmap'
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_od_scenario_comparison(data: Dict, od_source: str, output_dir: Path, dpi: int):
    """Plots #2-3: Grouped bar charts comparing Distance JSD across scenarios"""
    logger.info(f"    {2 if od_source=='train' else 3}. {od_source.upper()} OD comparison")
    
    scenarios = get_scenario_list(data, od_source, 'vanilla')
    scenarios = [s for s in scenarios if s in ['off_peak', 'peak', 'city_center', 
                                               'suburban', 'weekday', 'weekend']]
    models = ['vanilla', 'distilled', 'distilled_seed44']
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(scenarios))
    width = 0.25
    
    for i, model in enumerate(models):
        values = []
        for s in scenarios:
            val = get_metric_value(data, od_source, model, s, 'Distance_JSD')
            values.append(val if val is not None else 0)
        
        bars = ax.bar(x + i*width, values, width, label=MODEL_LABELS[model],
                     color=COLORS[model], alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Scenario', fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance JSD', fontsize=12, fontweight='bold')
    ax.set_title(f'{od_source.upper()} OD: Distance JSD by Scenario', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x + width)
    ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45, ha='right')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    output_path = output_dir / f'{od_source}_od_scenario_comparison'
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_metric_sensitivity(data: Dict, output_dir: Path, dpi: int):
    """Plot #8: 3×3 grid showing metric sensitivity to scenario types"""
    logger.info("    8. Metric sensitivity grid")
    
    # Define scenario groups
    temporal_scenarios = ['off_peak', 'peak', 'weekday', 'weekend']
    spatial_scenarios = ['city_center', 'suburban']
    trip_type_scenarios = ['to_center', 'from_center', 'within_center', 'suburban']
    
    metrics = ['Distance_JSD', 'Duration_JSD', 'Radius_JSD']
    metric_labels = ['Distance JSD', 'Duration JSD', 'Radius JSD']
    scenario_groups = [
        (temporal_scenarios, 'Temporal Scenarios'),
        (spatial_scenarios, 'Spatial Scenarios'),
        (trip_type_scenarios, 'Trip Types')
    ]
    
    models = ['vanilla', 'distilled', 'distilled_seed44']
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Metric Sensitivity by Scenario Type', fontsize=16, fontweight='bold', y=0.995)
    
    for row, (metric, label) in enumerate(zip(metrics, metric_labels)):
        for col, (scenarios, group_label) in enumerate(scenario_groups):
            ax = axes[row, col]
            
            # Filter scenarios that exist
            available_scenarios = [s for s in scenarios 
                                  if s in get_scenario_list(data, 'train', 'vanilla')]
            
            if not available_scenarios:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{label}\n{group_label}', fontsize=10)
                continue
            
            x = np.arange(len(available_scenarios))
            
            for model in models:
                values = []
                for s in available_scenarios:
                    val = get_metric_value(data, 'train', model, s, metric)
                    values.append(val if val is not None else np.nan)
                
                ax.plot(x, values, marker='o', label=MODEL_LABELS[model],
                       color=COLORS[model], linewidth=2, markersize=6)
            
            ax.set_xticks(x)
            ax.set_xticklabels([s.replace('_', ' ').title() for s in available_scenarios],
                              rotation=45, ha='right', fontsize=8)
            ax.set_ylabel(label, fontsize=9)
            ax.grid(alpha=0.3, linestyle='--')
            
            if row == 0:
                ax.set_title(group_label, fontsize=10, fontweight='bold')
            
            if row == 0 and col == 2:
                ax.legend(loc='upper right', fontsize=8, framealpha=0.95)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_path = output_dir / 'metric_sensitivity_by_scenario'
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches='tight')
    plt.close()

