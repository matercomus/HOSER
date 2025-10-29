"""
Application-focused plots.

Plots:
- #12: Application Use Case Radar Charts (3 radars)
- #13: Improvement Percentage Heatmap
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .data_loader import (
    get_metric_value,
    calculate_improvement,
    classify_models,
    get_model_colors,
    get_model_labels,
    get_available_scenarios,
    get_available_metrics,
    get_metric_display_labels,
)

logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


def plot_all(data: Dict, output_dir: Path, dpi: int = 300):
    """Generate all application plots"""
    logger.info("  🎯 Application plots...")
    
    plot_application_radars(data, output_dir, dpi)
    
    # Generate both individual and grid heatmaps
    plot_improvement_heatmaps_individual(data, output_dir, dpi)
    plot_improvement_heatmap_grid(data, output_dir, dpi)


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
    
    # Dynamically detect all models and get their colors/labels
    vanilla_models, distilled_models = classify_models(data, 'train')
    models = sorted(vanilla_models + distilled_models)
    model_colors = get_model_colors(data, 'train')
    model_labels = get_model_labels(data, 'train')
    
    if not models:
        logger.warning("    No models found for radar charts")
        return
    
    # Use first available scenario as representative
    scenarios = get_available_scenarios(data, 'train')
    scenario = scenarios[0] if scenarios else 'off_peak'
    
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
            for v, max_v in zip[tuple[Any, float]](values, max_vals):
                # Invert and normalize
                norm = 1 - min(v / max_v, 1.0)
                norm_values.append(norm)
            
            norm_values += norm_values[:1]  # Complete the circle
            
            ax.plot(angles, norm_values, 'o-', linewidth=2, label=model_labels[model],
                   color=model_colors[model])
            ax.fill(angles, norm_values, alpha=0.15, color=model_colors[model])
        
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


def plot_improvement_heatmaps_individual(data: Dict, output_dir: Path, dpi: int):
    """Plot #13a: Individual heatmaps for each distilled vs vanilla comparison"""
    logger.info("    13a. Individual improvement heatmaps")
    
    # Detect all vanilla and distilled models
    vanilla_models, distilled_models = classify_models(data, 'train')
    
    if not vanilla_models:
        logger.warning("    ⚠️  No vanilla models found, skipping individual heatmaps")
        return
    
    if not distilled_models:
        logger.warning("    ⚠️  No distilled models found, skipping individual heatmaps")
        return
    
    logger.info(f"    Generating {len(distilled_models)} × {len(vanilla_models)} = {len(distilled_models) * len(vanilla_models)} comparison heatmaps")
    
    # DYNAMIC: Extract scenarios and metrics from actual data
    scenarios = get_available_scenarios(data, 'train')
    metrics = get_available_metrics(data, 'train')
    metric_labels = get_metric_display_labels(metrics)
    
    # Validate we have data to plot
    if not scenarios:
        logger.warning("    ⚠️  No scenarios found in data, skipping individual heatmaps")
        return
    
    if not metrics:
        logger.warning("    ⚠️  No metrics found in data, skipping individual heatmaps")
        return
    
    logger.info(f"    Using {len(scenarios)} scenarios and {len(metrics)} metrics from data")
    
    # Generate one heatmap for each (distilled, vanilla) pair
    for distilled_model in distilled_models:
        for vanilla_model in vanilla_models:
            # Build improvement matrix
            improvement_matrix = np.zeros((len(scenarios), len(metrics)))
            
            for i, scenario in enumerate(scenarios):
                for j, metric in enumerate(metrics):
                    improvement = calculate_improvement(data, 'train', scenario, metric,
                                                       baseline=vanilla_model,
                                                       improved=distilled_model)
                    if improvement is not None:
                        improvement_matrix[i, j] = improvement
                    else:
                        improvement_matrix[i, j] = 0
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Calculate symmetric bounds using percentiles to exclude outliers
            # Use 5th and 95th percentiles for robust range estimation
            p5 = np.percentile(improvement_matrix, 5)
            p95 = np.percentile(improvement_matrix, 95)
            abs_max = max(abs(p5), abs(p95))
            
            # Handle case where all values are zero/None (avoid vmin==vmax)
            if abs_max < 0.01:
                abs_max = 10.0  # Use default range
            vmin, vmax = -abs_max, abs_max
            
            # Create red-white-green diverging colormap
            cmap = sns.diverging_palette(10, 130, s=80, l=55, as_cmap=True)
            
            # Create heatmap
            sns.heatmap(improvement_matrix, annot=True, fmt='.1f', cmap=cmap,
                       xticklabels=metric_labels,
                       yticklabels=[s.replace('_', ' ').title() for s in scenarios],
                       ax=ax, cbar_kws={'label': '% Improvement'}, 
                       center=0, vmin=vmin, vmax=vmax)
            
            # Create title with model names and average improvement
            distilled_label = distilled_model.replace('_', ' ').title()
            vanilla_label = vanilla_model.replace('_', ' ').title()
            
            # Calculate average improvement for title
            avg_improvement = np.mean(improvement_matrix[improvement_matrix > 0])
            avg_text = f' (Avg: {avg_improvement:.1f}%)' if not np.isnan(avg_improvement) else ''
            
            ax.set_title(f'Improvement of {distilled_label} over {vanilla_label}{avg_text}\n'
                        f'Across Scenarios and Metrics',
                        fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Metric', fontsize=11, fontweight='bold')
            ax.set_ylabel('Scenario', fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            
            # Save with descriptive filename
            output_path = output_dir / f'improvement_heatmap_{distilled_model}_vs_{vanilla_model}'
            plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches='tight')
            plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"      ✓ {distilled_model} vs {vanilla_model}")


def plot_improvement_heatmap_grid(data: Dict, output_dir: Path, dpi: int):
    """Plot #13b: Grid of heatmaps showing all model comparisons"""
    logger.info("    13b. Improvement heatmap grid (comprehensive overview)")
    
    # Detect all vanilla and distilled models
    vanilla_models, distilled_models = classify_models(data, 'train')
    
    if not vanilla_models:
        logger.warning("    ⚠️  No vanilla models found, skipping grid heatmap")
        return
    
    if not distilled_models:
        logger.warning("    ⚠️  No distilled models found, skipping grid heatmap")
        return
    
    n_vanilla = len(vanilla_models)
    n_distilled = len(distilled_models)
    
    logger.info(f"    Creating {n_distilled}×{n_vanilla} grid of comparison heatmaps")
    
    # DYNAMIC: Extract scenarios and metrics from actual data
    scenarios = get_available_scenarios(data, 'train')
    metrics = get_available_metrics(data, 'train')
    metric_labels = get_metric_display_labels(metrics)
    scenario_labels = [s.replace('_', ' ').title() for s in scenarios]
    
    # Validate we have data to plot
    if not scenarios:
        logger.warning("    ⚠️  No scenarios found in data, skipping grid heatmap")
        return
    
    if not metrics:
        logger.warning("    ⚠️  No metrics found in data, skipping grid heatmap")
        return
    
    logger.info(f"    Using {len(scenarios)} scenarios and {len(metrics)} metrics from data")
    
    # Calculate figure size: scale with number of subplots
    # Base size per subplot, with some adjustment for larger grids
    if n_vanilla <= 2 and n_distilled <= 2:
        fig_width = 8 * n_vanilla
        fig_height = 6 * n_distilled
        annot_fontsize = 8
    else:
        fig_width = 6 * n_vanilla
        fig_height = 5 * n_distilled
        annot_fontsize = 7
    
    fig, axes = plt.subplots(n_distilled, n_vanilla, 
                            figsize=(fig_width, fig_height),
                            squeeze=False)
    
    # Overall title
    fig.suptitle('Comprehensive Model Comparison: Distilled vs Vanilla Performance Improvement',
                fontsize=16, fontweight='bold', y=0.995)
    
    # Store all improvement values for shared colorbar
    all_improvements = []
    
    # First pass: calculate all improvements
    improvement_data = {}
    for i, distilled_model in enumerate(distilled_models):
        for j, vanilla_model in enumerate(vanilla_models):
            improvement_matrix = np.zeros((len(scenarios), len(metrics)))
            
            for s_idx, scenario in enumerate(scenarios):
                for m_idx, metric in enumerate(metrics):
                    improvement = calculate_improvement(data, 'train', scenario, metric,
                                                       baseline=vanilla_model,
                                                       improved=distilled_model)
                    if improvement is not None:
                        improvement_matrix[s_idx, m_idx] = improvement
                        all_improvements.append(improvement)
                    else:
                        improvement_matrix[s_idx, m_idx] = 0
            
            improvement_data[(i, j)] = improvement_matrix
    
    # Determine colorbar range using percentiles to exclude outliers
    if all_improvements:
        # Use 5th and 95th percentiles for robust range estimation
        p5 = np.percentile(all_improvements, 5)
        p95 = np.percentile(all_improvements, 95)
        abs_max = max(abs(p5), abs(p95))
        
        # Handle case where all values are zero/None (avoid vmin==vmax)
        if abs_max < 0.01:
            abs_max = 10.0  # Use default range
        vmin, vmax = -abs_max, abs_max
    else:
        vmin, vmax = -100, 100
    
    # Create red-white-green diverging colormap
    cmap = sns.diverging_palette(10, 130, s=80, l=55, as_cmap=True)
    
    # Second pass: create heatmaps
    for i, distilled_model in enumerate(distilled_models):
        for j, vanilla_model in enumerate(vanilla_models):
            ax = axes[i, j]
            improvement_matrix = improvement_data[(i, j)]
            
            # Create heatmap
            # Only show colorbar on rightmost column
            show_cbar = (j == n_vanilla - 1)
            
            sns.heatmap(improvement_matrix, annot=True, fmt='.1f', cmap=cmap,
                       xticklabels=metric_labels,
                       yticklabels=scenario_labels if j == 0 else [],
                       ax=ax, cbar=show_cbar,
                       cbar_kws={'label': '% Improvement'} if show_cbar else {},
                       center=0, vmin=vmin, vmax=vmax,
                       annot_kws={'fontsize': annot_fontsize})
            
            # Subplot title with average improvement
            distilled_label = distilled_model.replace('_', ' ').title()
            vanilla_label = vanilla_model.replace('_', ' ').title()
            
            # Calculate average improvement for title
            avg_improvement = np.mean(improvement_matrix[improvement_matrix > 0])
            avg_text = f' ({avg_improvement:.1f}%)' if not np.isnan(avg_improvement) else ''
            
            title_fontsize = 10 if n_vanilla > 2 or n_distilled > 2 else 11
            ax.set_title(f'{distilled_label} vs {vanilla_label}{avg_text}',
                        fontsize=title_fontsize, fontweight='bold', pad=8)
            
            # Labels
            if i == n_distilled - 1:  # Bottom row
                ax.set_xlabel('Metric', fontsize=9, fontweight='bold')
            else:
                ax.set_xlabel('')
            
            if j == 0:  # Leftmost column
                ax.set_ylabel('Scenario', fontsize=9, fontweight='bold')
            else:
                ax.set_ylabel('')
            
            # Adjust tick label sizes
            ax.tick_params(axis='both', labelsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save
    output_path = output_dir / 'improvement_heatmap_grid'
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches='tight')
    plt.close()
    
    logger.info(f"      ✓ Grid saved with {n_distilled * n_vanilla} comparisons")


def plot_improvement_heatmap(data: Dict, output_dir: Path, dpi: int):
    """Plot #13: Heatmap showing % improvement over vanilla (DEPRECATED - kept for compatibility)"""
    logger.info("    13. Improvement percentage heatmap (legacy single comparison)")
    
    # Use dynamic extraction even for legacy function
    scenarios = get_available_scenarios(data, 'train')
    metrics = get_available_metrics(data, 'train')
    metric_labels = get_metric_display_labels(metrics)
    
    if not scenarios or not metrics:
        logger.warning("    ⚠️  No scenarios/metrics found, skipping legacy heatmap")
        return
    
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
    
    # Calculate symmetric bounds using percentiles to exclude outliers
    # Use 5th and 95th percentiles for robust range estimation
    p5 = np.percentile(improvement_matrix, 5)
    p95 = np.percentile(improvement_matrix, 95)
    abs_max = max(abs(p5), abs(p95))
    
    # Handle case where all values are zero/None (avoid vmin==vmax)
    if abs_max < 0.01:
        abs_max = 10.0  # Use default range
    vmin, vmax = -abs_max, abs_max
    
    # Create red-white-green diverging colormap
    cmap = sns.diverging_palette(10, 130, s=80, l=55, as_cmap=True)
    
    # Create heatmap
    sns.heatmap(improvement_matrix, annot=True, fmt='.1f', cmap=cmap,
               xticklabels=metric_labels,
               yticklabels=[s.replace('_', ' ').title() for s in scenarios],
               ax=ax, cbar_kws={'label': '% Improvement'}, 
               center=0, vmin=vmin, vmax=vmax)
    
    # Calculate average improvement for title
    avg_improvement = np.mean(improvement_matrix[improvement_matrix > 0])
    avg_text = f' (Avg: {avg_improvement:.1f}%)' if not np.isnan(avg_improvement) else ''
    
    ax.set_title(f'Improvement of Distilled (seed 44) over Vanilla{avg_text}\n'
                f'Across Scenarios and Metrics',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Metric', fontsize=11, fontweight='bold')
    ax.set_ylabel('Scenario', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / 'improvement_heatmap'
    plt.savefig(f"{output_path}.png", dpi=dpi, bbox_inches='tight')
    plt.savefig(f"{output_path}.pdf", dpi=dpi, bbox_inches='tight')
    plt.close()

