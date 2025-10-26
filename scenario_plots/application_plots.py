"""
Application-focused plots.

Plots:
- #12: Application Use Case Radar Charts (3 radars)
- #13: Improvement Percentage Heatmap
"""

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .data_loader import get_metric_value, calculate_improvement

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
    """Generate all application plots"""
    logger.info("  🎯 Application plots...")
    
    plot_application_radars(data, output_dir, dpi)
    plot_improvement_heatmap(data, output_dir, dpi)


def plot_application_radars(data: Dict, output_dir: Path, dpi: int):
    """Plot #12: 3 radar charts for different applications"""
    logger.info("    12. Application use case radars")
    
    # Define metrics for each application
    applications = {
        'Route Planning': {
            'metrics': ['Distance_JSD', 'Radius_JSD', 'Hausdorff_km', 'DTW_km'],
            'labels': ['Distance\nQuality', 'Spatial\nComplexity', 'Route\nDeviation', 'Path\nSimilarity'],
            'description': 'Emphasis on spatial accuracy and route quality'
        },
        'Traffic Simulation': {
            'metrics': ['Distance_JSD', 'Duration_JSD', 'DTW_km', 'EDR'],
            'labels': ['Distance\nRealism', 'Duration\nRealism', 'Temporal\nAlignment', 'Sequence\nSimilarity'],
            'description': 'Focus on realistic distributions and temporal patterns'
        },
        'Urban Planning': {
            'metrics': ['Distance_JSD', 'Radius_JSD', 'Duration_JSD', 'Hausdorff_km'],
            'labels': ['Trip Length\nPatterns', 'Spatial\nSpread', 'Time\nPatterns', 'Coverage\nArea'],
            'description': 'Aggregate patterns and spatial coverage'
        }
    }
    
    models = ['vanilla', 'distilled', 'distilled_seed44']
    
    # Use 'off_peak' scenario as representative
    scenario = 'off_peak'
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))
    fig.suptitle('Application-Specific Performance Profiles', fontsize=16, fontweight='bold')
    
    for idx, (app_name, app_config) in enumerate(applications.items()):
        ax = axes[idx]
        metrics = app_config['metrics']
        labels = app_config['labels']
        
        # Number of variables
        num_vars = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each model
        for model in models:
            values = []
            for metric in metrics:
                val = get_metric_value(data, 'train', model, scenario, metric)
                if val is not None:
                    values.append(val)
                else:
                    values.append(0)
            
            # Normalize values (invert so higher is better)
            # For JSD, Hausdorff, DTW, EDR: lower is better, so invert
            max_vals = [0.3, 0.3, 2.0, 50.0]  # Approximate max reasonable values
            norm_values = []
            for v, max_v in zip(values, max_vals):
                # Invert and normalize
                norm = 1 - min(v / max_v, 1.0)
                norm_values.append(norm)
            
            norm_values += norm_values[:1]  # Complete the circle
            
            ax.plot(angles, norm_values, 'o-', linewidth=2, label=MODEL_LABELS[model],
                   color=COLORS[model])
            ax.fill(angles, norm_values, alpha=0.15, color=COLORS[model])
        
        # Fix axis
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.set_title(f'{app_name}\n{app_config["description"]}', 
                    fontsize=11, fontweight='bold', pad=20)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if idx == 2:
            ax.legend(loc='upper left', bbox_to_anchor=(1.2, 1.0), framealpha=0.95)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = output_dir / 'application_use_case_radar'
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_improvement_heatmap(data: Dict, output_dir: Path, dpi: int):
    """Plot #13: Heatmap showing % improvement over vanilla"""
    logger.info("    13. Improvement percentage heatmap")
    
    scenarios = ['off_peak', 'peak', 'city_center', 'suburban', 'weekday', 'weekend']
    metrics = ['Distance_JSD', 'Duration_JSD', 'Radius_JSD', 
               'Hausdorff_km', 'DTW_km', 'EDR']
    metric_labels = ['Distance\nJSD', 'Duration\nJSD', 'Radius\nJSD',
                     'Hausdorff\n(km)', 'DTW\n(km)', 'EDR']
    
    # Build improvement matrix
    improvement_matrix = np.zeros((len(scenarios), len(metrics)))
    
    for i, scenario in enumerate(scenarios):
        for j, metric in enumerate(metrics):
            improvement = calculate_improvement(data, 'train', scenario, metric,
                                               baseline='vanilla',
                                               improved='distilled_seed44')
            if improvement is not None:
                improvement_matrix[i, j] = improvement
            else:
                improvement_matrix[i, j] = 0
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(improvement_matrix, annot=True, fmt='.1f', cmap='Greens',
               xticklabels=metric_labels,
               yticklabels=[s.replace('_', ' ').title() for s in scenarios],
               ax=ax, cbar_kws={'label': '% Improvement'}, 
               vmin=0, vmax=100)
    
    ax.set_title('Improvement of Distilled (seed 44) over Vanilla\nAcross Scenarios and Metrics',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Metric', fontsize=11, fontweight='bold')
    ax.set_ylabel('Scenario', fontsize=11, fontweight='bold')
    
    # Add average improvement annotation
    avg_improvement = np.mean(improvement_matrix[improvement_matrix > 0])
    ax.text(0.02, 0.98, f'Average Improvement: {avg_improvement:.1f}%',
           transform=ax.transAxes, ha='left', va='top',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
           fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / 'improvement_heatmap'
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches='tight')
    plt.close()

