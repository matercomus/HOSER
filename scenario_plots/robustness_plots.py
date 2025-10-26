"""
Robustness and difficulty analysis plots.

Plots:
- #6: Seed Robustness Across Scenarios (6-panel bars)
- #7: Scenario Difficulty Ranking (horizontal bars)
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


def plot_all(data: Dict, output_dir: Path, dpi: int = 300):
    """Generate all robustness plots"""
    logger.info("  🔬 Robustness plots...")
    
    plot_seed_robustness(data, output_dir, dpi)
    plot_scenario_difficulty_ranking(data, output_dir, dpi)


def plot_seed_robustness(data: Dict, output_dir: Path, dpi: int):
    """Plot #6: 6-panel bar charts comparing seeds"""
    logger.info("    6. Seed robustness across scenarios")
    
    scenarios = get_scenario_list(data, 'train', 'distilled')
    scenarios = [s for s in scenarios if s in ['off_peak', 'peak', 'city_center', 
                                               'suburban', 'weekday', 'weekend']]
    
    metrics = ['Distance_JSD', 'Duration_JSD', 'Radius_JSD', 
               'Hausdorff_km', 'DTW_km', 'EDR']
    metric_labels = ['Distance JSD', 'Duration JSD', 'Radius JSD',
                     'Hausdorff (km)', 'DTW (km)', 'EDR']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Seed Robustness: Distilled Model Comparison', fontsize=16, fontweight='bold')
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 3, idx % 3]
        x = np.arange(len(scenarios))
        width = 0.35
        
        # Seed 42
        seed42_vals = [get_metric_value(data, 'train', 'distilled', s, metric)
                       for s in scenarios]
        seed42_vals = [v if v is not None else 0 for v in seed42_vals]
        
        # Seed 44
        seed44_vals = [get_metric_value(data, 'train', 'distilled_seed44', s, metric)
                       for s in scenarios]
        seed44_vals = [v if v is not None else 0 for v in seed44_vals]
        
        bars1 = ax.bar(x - width/2, seed42_vals, width, label='Seed 42',
                      color='#3498db', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, seed44_vals, width, label='Seed 44',
                      color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Calculate CV (Coefficient of Variation)
        cv_values = []
        for v1, v2 in zip(seed42_vals, seed44_vals):
            if v1 > 0 and v2 > 0:
                mean = (v1 + v2) / 2
                std = np.std([v1, v2])
                cv = (std / mean) * 100
                cv_values.append(cv)
        
        if cv_values:
            mean_cv = np.mean(cv_values)
            ax.text(0.98, 0.98, f'Avg CV: {mean_cv:.1f}%', 
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)
        
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', ' ').title() for s in scenarios],
                          rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Value', fontsize=10)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    output_path = output_dir / 'seed_robustness_scenarios'
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_scenario_difficulty_ranking(data: Dict, output_dir: Path, dpi: int):
    """Plot #7: Horizontal bar chart ranking scenarios by difficulty"""
    logger.info("    7. Scenario difficulty ranking")
    
    scenarios = get_scenario_list(data, 'train', 'distilled_seed44')
    
    # Calculate difficulty score: average of normalized Distance JSD and Radius JSD
    difficulty_scores = {}
    scenario_types = {}
    
    # Get all values for normalization
    all_dist_jsd = []
    all_radius_jsd = []
    
    for s in scenarios:
        dist = get_metric_value(data, 'train', 'distilled_seed44', s, 'Distance_JSD')
        radius = get_metric_value(data, 'train', 'distilled_seed44', s, 'Radius_JSD')
        if dist is not None:
            all_dist_jsd.append(dist)
        if radius is not None:
            all_radius_jsd.append(radius)
    
    min_dist, max_dist = min(all_dist_jsd), max(all_dist_jsd)
    min_radius, max_radius = min(all_radius_jsd), max(all_radius_jsd)
    
    for s in scenarios:
        dist = get_metric_value(data, 'train', 'distilled_seed44', s, 'Distance_JSD')
        radius = get_metric_value(data, 'train', 'distilled_seed44', s, 'Radius_JSD')
        
        if dist is not None and radius is not None:
            # Normalize to 0-1
            norm_dist = (dist - min_dist) / (max_dist - min_dist) if max_dist > min_dist else 0
            norm_radius = (radius - min_radius) / (max_radius - min_radius) if max_radius > min_radius else 0
            
            # Average difficulty
            difficulty_scores[s] = (norm_dist + norm_radius) / 2
            
            # Categorize scenario type
            if s in ['off_peak', 'peak', 'weekday', 'weekend']:
                scenario_types[s] = 'temporal'
            elif s in ['city_center', 'suburban']:
                scenario_types[s] = 'spatial'
            else:
                scenario_types[s] = 'trip_type'
    
    # Sort by difficulty
    sorted_scenarios = sorted(difficulty_scores.items(), key=lambda x: x[1])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(sorted_scenarios))
    difficulties = [v for _, v in sorted_scenarios]
    labels = [k.replace('_', ' ').title() for k, _ in sorted_scenarios]
    
    # Color by type
    colors = []
    for k, _ in sorted_scenarios:
        stype = scenario_types.get(k, 'other')
        if stype == 'temporal':
            colors.append('#3498db')
        elif stype == 'spatial':
            colors.append('#e67e22')
        else:
            colors.append('#9b59b6')
    
    bars = ax.barh(y_pos, difficulties, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, difficulties)):
        ax.text(val, i, f' {val:.3f}', va='center', fontsize=9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Difficulty Score (normalized)', fontsize=11, fontweight='bold')
    ax.set_title('Scenario Difficulty Ranking for Distilled Model (seed 44)',
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Temporal'),
        Patch(facecolor='#e67e22', label='Spatial'),
        Patch(facecolor='#9b59b6', label='Trip Type')
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95)
    
    plt.tight_layout()
    
    output_path = output_dir / 'scenario_difficulty_ranking'
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches='tight')
    plt.close()

