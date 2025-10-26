"""
Advanced analysis plots.

Plots:
- #9: Duration Ceiling Effect (box plots)
- #10: Spatial Metrics Differentiation (scatter with zones)
- #11: Scenario Variance Analysis (range plot)
"""

import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .data_loader import get_metric_value, get_scenario_list

logger = logging.getLogger(__name__)

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
    """Generate all analysis plots"""
    logger.info("  📊 Analysis plots...")
    
    plot_duration_ceiling(data, output_dir, dpi)
    plot_spatial_differentiation(data, output_dir, dpi)
    plot_variance_analysis(data, output_dir, dpi)


def plot_duration_ceiling(data: Dict, output_dir: Path, dpi: int):
    """Plot #9: Box plots showing duration JSD ceiling effect"""
    logger.info("    9. Duration ceiling effect")
    
    # Group scenarios by type
    scenario_groups = {
        'Temporal': ['off_peak', 'peak', 'weekday', 'weekend'],
        'Spatial': ['city_center', 'suburban'],
        'Trip Type': ['to_center', 'from_center', 'within_center']
    }
    
    models = ['vanilla', 'distilled', 'distilled_seed44']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    all_data = []
    all_labels = []
    positions = []
    pos = 0
    
    for group_name, scenarios in scenario_groups.items():
        for model in models:
            values = []
            for s in scenarios:
                if s in get_scenario_list(data, 'train', model):
                    val = get_metric_value(data, 'train', model, s, 'Duration_JSD')
                    if val is not None:
                        values.append(val)
            
            if values:
                all_data.append(values)
                all_labels.append(f"{group_name}\n{MODEL_LABELS[model]}")
                positions.append(pos)
                pos += 1
        
        pos += 0.5  # Gap between groups
    
    # Create box plots
    bp = ax.boxplot(all_data, positions=positions, widths=0.6, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=6, alpha=0.5))
    
    # Overlay individual points
    for i, (values, pos) in enumerate(zip(all_data, positions)):
        x = np.random.normal(pos, 0.04, size=len(values))
        ax.scatter(x, values, alpha=0.4, s=30, color='navy')
    
    # Horizontal line at "excellent" threshold
    ax.axhline(y=0.020, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Excellent (<0.020)')
    
    ax.set_xticks(positions)
    ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Duration JSD', fontsize=11, fontweight='bold')
    ax.set_title('Duration JSD Ceiling Effect Across Scenario Types',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 0.05)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    output_path = output_dir / 'duration_ceiling_effect'
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_spatial_differentiation(data: Dict, output_dir: Path, dpi: int):
    """Plot #10: Scatter plot with performance zones"""
    logger.info("    10. Spatial metrics differentiation")
    
    scenarios = get_scenario_list(data, 'train', 'vanilla')
    models = ['vanilla', 'distilled', 'distilled_seed44']
    
    # Scenario markers
    scenario_markers = {
        'off_peak': 'o', 'peak': 's', 'weekday': '^', 'weekend': 'v',
        'city_center': 'D', 'suburban': 'p', 'to_center': '*',
        'from_center': 'h', 'within_center': '+'
    }
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw performance zones
    ax.axhspan(0, 0.03, alpha=0.1, color='green', label='Excellent (<0.03)')
    ax.axhspan(0.03, 0.10, alpha=0.1, color='yellow', label='Good (0.03-0.10)')
    ax.axhspan(0.10, 0.35, alpha=0.1, color='red', label='Poor (>0.10)')
    
    for model in models:
        for scenario in scenarios:
            dist_jsd = get_metric_value(data, 'train', model, scenario, 'Distance_JSD')
            radius_jsd = get_metric_value(data, 'train', model, scenario, 'Radius_JSD')
            
            if dist_jsd is not None and radius_jsd is not None:
                marker = scenario_markers.get(scenario, 'o')
                ax.scatter(dist_jsd, radius_jsd, s=150, marker=marker,
                          color=COLORS[model], alpha=0.7, edgecolors='black', linewidth=1.5,
                          label=f'{MODEL_LABELS[model]}' if scenario == scenarios[0] else '')
    
    ax.set_xlabel('Distance JSD', fontsize=11, fontweight='bold')
    ax.set_ylabel('Radius of Gyration JSD', fontsize=11, fontweight='bold')
    ax.set_title('Spatial Metrics Differentiation: Distance vs Radius JSD',
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(alpha=0.3, linestyle='--')
    
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', framealpha=0.95, fontsize=9)
    
    # Add diagonal reference line
    max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1, label='y=x')
    
    plt.tight_layout()
    
    output_path = output_dir / 'spatial_metrics_differentiation'
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_variance_analysis(data: Dict, output_dir: Path, dpi: int):
    """Plot #11: Range plot showing variance across scenarios"""
    logger.info("    11. Scenario variance analysis")
    
    scenarios = get_scenario_list(data, 'train', 'vanilla')
    metrics = ['Distance_JSD', 'Duration_JSD', 'Radius_JSD', 
               'Hausdorff_km', 'DTW_km', 'EDR']
    metric_labels = ['Distance\nJSD', 'Duration\nJSD', 'Radius\nJSD',
                     'Hausdorff\n(km)', 'DTW\n(km)', 'EDR']
    models = ['vanilla', 'distilled', 'distilled_seed44']
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    metric_centers = []
    pos = 0
    
    for metric, label in zip(metrics, metric_labels):
        metric_start = pos
        
        for model in models:
            values = []
            for s in scenarios:
                val = get_metric_value(data, 'train', model, s, metric)
                if val is not None:
                    values.append(val)
            
            if values:
                # Normalize to 0-1 for this metric
                min_val, max_val = min(values), max(values)
                if max_val > min_val:
                    norm_values = [(v - min_val) / (max_val - min_val) for v in values]
                else:
                    norm_values = [0.5] * len(values)
                
                mean_val = np.mean(norm_values)
                min_norm = min(norm_values)
                max_norm = max(norm_values)
                std_val = np.std(norm_values)
                cv = (std_val / mean_val * 100) if mean_val > 0 else 0
                
                # Plot range
                ax.plot([pos, pos], [min_norm, max_norm], color=COLORS[model], linewidth=3, alpha=0.7)
                ax.scatter(pos, mean_val, s=150, color=COLORS[model], zorder=5, 
                          edgecolors='black', linewidths=1.5)
                
                # Add CV annotation
                ax.text(pos, max_norm + 0.03, f'{cv:.1f}%', ha='center', fontsize=8,
                       color=COLORS[model], fontweight='bold')
                
                pos += 0.3
        
        # Store center position for this metric
        metric_center = (metric_start + pos - 0.3) / 2
        metric_centers.append(metric_center)
        
        pos += 0.5  # Gap between metrics
    
    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS[m],
                             markersize=10, label=MODEL_LABELS[m], markeredgecolor='black')
                      for m in models]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.95, fontsize=10)
    
    ax.set_xticks(metric_centers)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel('Normalized Value (0-1)', fontsize=11, fontweight='bold')
    ax.set_title('Scenario Variance Analysis: Min/Mean/Max Across All Scenarios',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    output_path = output_dir / 'scenario_variance_analysis'
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches='tight')
    plt.close()

