#!/usr/bin/env python3
"""
Create comprehensive analysis figures for HOSER distillation evaluation.

This script generates publication-quality PDF figures comparing vanilla vs distilled models.
All figures are saved to the figures/ directory.

Usage:
    uv run python create_analysis_figures.py
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif',
    'text.usetex': False,  # Set to True if LaTeX is available
})

# Color scheme
COLORS = {
    'distilled': '#2ecc71',      # Green
    'distilled_seed44': '#27ae60',  # Dark green
    'vanilla': '#e74c3c',        # Red
    'real': '#34495e',           # Dark gray
}

MARKERS = {
    'distilled': 'o',
    'distilled_seed44': 's',
    'vanilla': '^',
}


class EvaluationVisualizer:
    """Create visualizations for HOSER evaluation results"""
    
    def __init__(self, eval_dir: str = "eval", figures_dir: str = "figures"):
        self.eval_dir = Path(eval_dir)
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(exist_ok=True)
        
        # Load all results
        self.results = self._load_results()
        print(f"📊 Loaded {len(self.results)} evaluation results")
    
    def _load_results(self) -> List[Dict]:
        """Load all results.json files from eval directory"""
        results = []
        for results_file in self.eval_dir.glob("*/results.json"):
            with open(results_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        return results
    
    def _parse_model_info(self, result: Dict) -> Tuple[str, str]:
        """Extract model type and OD source from result metadata"""
        metadata = result['metadata']
        
        # Determine OD source
        od_source = metadata.get('od_source', 'unknown')
        
        # Determine model type from generated file path
        gen_file = metadata.get('generated_file', '')
        if 'distilled_seed44' in gen_file:
            model_type = 'distilled_seed44'
        elif 'distilled' in gen_file:
            model_type = 'distilled'
        elif 'vanilla' in gen_file:
            model_type = 'vanilla'
        else:
            model_type = 'unknown'
        
        return model_type, od_source
    
    def create_all_figures(self):
        """Generate all analysis figures"""
        print("\n🎨 Creating analysis figures...")
        
        self.plot_distance_distributions()
        self.plot_od_matching_rates()
        self.plot_jsd_comparison()
        self.plot_metrics_heatmap()
        self.plot_train_test_comparison()
        self.plot_seed_robustness()
        self.plot_local_metrics()
        self.plot_performance_radar()
        
        print(f"\n✅ All figures saved to {self.figures_dir}/")
    
    def plot_distance_distributions(self):
        """Figure 1: Distance distribution comparison"""
        print("  📈 Creating distance distributions...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Train OD
        for result in self.results:
            model_type, od_source = self._parse_model_info(result)
            if od_source == 'train':
                dist_mean = result['Distance_gen_mean']
                jsd = result['Distance_JSD']
                
                ax1.bar(model_type, dist_mean, 
                       color=COLORS.get(model_type, 'gray'),
                       alpha=0.8, label=f"{model_type}\n(JSD={jsd:.4f})")
        
        ax1.axhline(y=5.16, color=COLORS['real'], linestyle='--', 
                   linewidth=2, label='Real Average')
        ax1.set_ylabel('Average Distance (km)')
        ax1.set_title('Train OD Pairs: Generated Distance vs Real')
        ax1.legend(loc='upper right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Test OD
        for result in self.results:
            model_type, od_source = self._parse_model_info(result)
            if od_source == 'test':
                dist_mean = result['Distance_gen_mean']
                jsd = result['Distance_JSD']
                
                ax2.bar(model_type, dist_mean,
                       color=COLORS.get(model_type, 'gray'),
                       alpha=0.8, label=f"{model_type}\n(JSD={jsd:.4f})")
        
        ax2.axhline(y=5.16, color=COLORS['real'], linestyle='--',
                   linewidth=2, label='Real Average')
        ax2.set_ylabel('Average Distance (km)')
        ax2.set_title('Test OD Pairs: Generated Distance vs Real')
        ax2.legend(loc='upper right')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'distance_distributions.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(self.figures_dir / 'distance_distributions.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_od_matching_rates(self):
        """Figure 2: OD pair matching coverage"""
        print("  📊 Creating OD matching rates...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Organize data
        data = {}
        for result in self.results:
            model_type, od_source = self._parse_model_info(result)
            matched = result['matched_od_pairs']
            total = result['total_generated_od_pairs']
            rate = (matched / total) * 100
            
            key = f"{model_type}_{od_source}"
            data[key] = {'matched': matched, 'total': total, 'rate': rate,
                        'model': model_type, 'od': od_source}
        
        # Create grouped bar chart
        x_pos = np.arange(len(data))
        bar_width = 0.35
        
        # Split by OD source
        train_data = {k: v for k, v in data.items() if v['od'] == 'train'}
        test_data = {k: v for k, v in data.items() if v['od'] == 'test'}
        
        # Plot
        models = ['distilled', 'distilled_seed44', 'vanilla']
        x_pos = np.arange(len(models))
        
        train_rates = [train_data.get(f"{m}_train", {}).get('rate', 0) for m in models]
        test_rates = [test_data.get(f"{m}_test", {}).get('rate', 0) for m in models]
        
        ax.bar(x_pos - bar_width/2, train_rates, bar_width,
              label='Train OD', alpha=0.8, color='#3498db')
        ax.bar(x_pos + bar_width/2, test_rates, bar_width,
              label='Test OD', alpha=0.8, color='#9b59b6')
        
        # Add value labels on bars
        for i, (train, test) in enumerate(zip(train_rates, test_rates)):
            ax.text(i - bar_width/2, train + 2, f'{train:.1f}%',
                   ha='center', va='bottom', fontsize=9)
            ax.text(i + bar_width/2, test + 2, f'{test:.1f}%',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('OD Pair Match Rate (%)')
        ax.set_title('OD Pair Coverage: Percentage of Generated OD Pairs Matching Real Data')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'od_matching_rates.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(self.figures_dir / 'od_matching_rates.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_jsd_comparison(self):
        """Figure 3: JSD comparison across metrics"""
        print("  📉 Creating JSD comparison...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        models = ['distilled', 'distilled_seed44', 'vanilla']
        
        # Distance JSD
        for od_source in ['train', 'test']:
            values = []
            for model in models:
                for result in self.results:
                    m, o = self._parse_model_info(result)
                    if m == model and o == od_source:
                        values.append(result['Distance_JSD'])
                        break
                else:
                    values.append(0)
            
            x_pos = np.arange(len(models))
            offset = -0.2 if od_source == 'train' else 0.2
            ax1.bar(x_pos + offset, values, 0.35,
                   label=f'{od_source.title()} OD',
                   alpha=0.8)
        
        ax1.set_ylabel('Distance JSD (lower is better)')
        ax1.set_title('Distance Distribution Quality')
        ax1.set_xticks(range(len(models)))
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.set_yscale('log')
        ax1.grid(axis='y', alpha=0.3)
        
        # Radius JSD
        for od_source in ['train', 'test']:
            values = []
            for model in models:
                for result in self.results:
                    m, o = self._parse_model_info(result)
                    if m == model and o == od_source:
                        values.append(result['Radius_JSD'])
                        break
                else:
                    values.append(0)
            
            x_pos = np.arange(len(models))
            offset = -0.2 if od_source == 'train' else 0.2
            ax2.bar(x_pos + offset, values, 0.35,
                   label=f'{od_source.title()} OD',
                   alpha=0.8)
        
        ax2.set_ylabel('Radius JSD (lower is better)')
        ax2.set_title('Radius of Gyration Distribution Quality')
        ax2.set_xticks(range(len(models)))
        ax2.set_xticklabels(models)
        ax2.legend()
        ax2.set_yscale('log')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'jsd_comparison.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(self.figures_dir / 'jsd_comparison.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_metrics_heatmap(self):
        """Figure 4: Comprehensive metrics heatmap"""
        print("  🔥 Creating metrics heatmap...")
        
        # Prepare data matrix
        models_od = []
        metrics_data = []
        metric_names = ['Distance_JSD', 'Radius_JSD', 'Duration_JSD',
                       'Hausdorff_km', 'DTW_km', 'EDR']
        
        for result in self.results:
            model_type, od_source = self._parse_model_info(result)
            models_od.append(f"{model_type}\n({od_source})")
            
            row = []
            for metric in metric_names:
                row.append(result.get(metric, 0))
            metrics_data.append(row)
        
        # Normalize each metric to 0-1 scale for visualization
        data_array = np.array(metrics_data)
        normalized_data = np.zeros_like(data_array)
        for i in range(data_array.shape[1]):
            col = data_array[:, i]
            normalized_data[:, i] = (col - col.min()) / (col.max() - col.min() + 1e-10)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(normalized_data.T, cmap='RdYlGn_r', aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(len(models_od)))
        ax.set_yticks(np.arange(len(metric_names)))
        ax.set_xticklabels(models_od, fontsize=9)
        ax.set_yticklabels(metric_names)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Score\n(0=best, 1=worst for this metric)', rotation=270, labelpad=20)
        
        # Add value annotations
        for i in range(len(models_od)):
            for j in range(len(metric_names)):
                ax.text(i, j, f'{data_array[i, j]:.3f}',
                       ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Comprehensive Metrics Comparison (Normalized)')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'metrics_heatmap.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(self.figures_dir / 'metrics_heatmap.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_train_test_comparison(self):
        """Figure 5: Train vs test performance (generalization)"""
        print("  🎯 Creating train vs test comparison...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        metrics = ['Distance_JSD', 'Radius_JSD', 'matched_od_pairs', 'Distance_gen_mean']
        titles = ['Distance JSD', 'Radius JSD', 'Matched OD Pairs', 'Generated Distance (km)']
        
        models = ['distilled', 'distilled_seed44', 'vanilla']
        x_pos = np.arange(len(models))
        bar_width = 0.35
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx]
            
            train_values = []
            test_values = []
            
            for model in models:
                train_val = None
                test_val = None
                
                for result in self.results:
                    m, o = self._parse_model_info(result)
                    if m == model:
                        if o == 'train':
                            train_val = result.get(metric, 0)
                        elif o == 'test':
                            test_val = result.get(metric, 0)
                
                train_values.append(train_val if train_val is not None else 0)
                test_values.append(test_val if test_val is not None else 0)
            
            ax.bar(x_pos - bar_width/2, train_values, bar_width,
                  label='Train OD', alpha=0.8, color='#3498db')
            ax.bar(x_pos + bar_width/2, test_values, bar_width,
                  label='Test OD', alpha=0.8, color='#e67e22')
            
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add improvement indicators for JSD metrics
            if 'JSD' in metric:
                for i, (train, test) in enumerate(zip(train_values, test_values)):
                    if test < train:
                        ax.annotate('', xy=(i + bar_width/2, test), 
                                  xytext=(i - bar_width/2, train),
                                  arrowprops=dict(arrowstyle='->', color='green', lw=2))
        
        plt.suptitle('Generalization: Train vs Test Performance', fontsize=16, y=1.00)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'train_test_comparison.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(self.figures_dir / 'train_test_comparison.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_seed_robustness(self):
        """Figure 6: Robustness across seeds"""
        print("  🔄 Creating seed robustness comparison...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        metrics = ['Distance_JSD', 'Radius_JSD', 'matched_od_pairs', 'Distance_gen_mean']
        titles = ['Distance JSD', 'Radius JSD', 'Matched OD Pairs', 'Generated Distance (km)']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx]
            
            # Get values for both distilled seeds
            for od_source in ['train', 'test']:
                seed42_val = None
                seed44_val = None
                
                for result in self.results:
                    m, o = self._parse_model_info(result)
                    if o == od_source:
                        if m == 'distilled':
                            seed42_val = result.get(metric, 0)
                        elif m == 'distilled_seed44':
                            seed44_val = result.get(metric, 0)
                
                # Plot line connecting seeds
                if seed42_val is not None and seed44_val is not None:
                    ax.plot([0, 1], [seed42_val, seed44_val],
                           'o-', label=f'{od_source.title()} OD',
                           linewidth=2, markersize=10, alpha=0.7)
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Seed 42', 'Seed 44'])
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Calculate and display coefficient of variation
            if seed42_val is not None and seed44_val is not None:
                mean_val = (seed42_val + seed44_val) / 2
                std_val = np.std([seed42_val, seed44_val])
                cv = (std_val / mean_val) * 100 if mean_val > 0 else 0
                ax.text(0.5, 0.95, f'CV: {cv:.1f}%',
                       transform=ax.transAxes, ha='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Robustness Across Seeds (Distilled Model)', fontsize=16, y=1.00)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'seed_robustness.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(self.figures_dir / 'seed_robustness.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_local_metrics(self):
        """Figure 7: Local trajectory-level metrics"""
        print("  📍 Creating local metrics comparison...")
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        local_metrics = ['Hausdorff_km', 'DTW_km', 'EDR']
        titles = ['Hausdorff Distance (km)', 'DTW Distance (km)', 'EDR (normalized)']
        
        models = ['distilled', 'distilled_seed44', 'vanilla']
        x_pos = np.arange(len(models))
        bar_width = 0.35
        
        for idx, (metric, title) in enumerate(zip(local_metrics, titles)):
            ax = axes[idx]
            
            train_values = []
            test_values = []
            
            for model in models:
                train_val = None
                test_val = None
                
                for result in self.results:
                    m, o = self._parse_model_info(result)
                    if m == model:
                        if o == 'train':
                            train_val = result.get(metric, 0)
                        elif o == 'test':
                            test_val = result.get(metric, 0)
                
                train_values.append(train_val if train_val is not None else 0)
                test_values.append(test_val if test_val is not None else 0)
            
            ax.bar(x_pos - bar_width/2, train_values, bar_width,
                  label='Train OD', alpha=0.8, color='#3498db')
            ax.bar(x_pos + bar_width/2, test_values, bar_width,
                  label='Test OD', alpha=0.8, color='#e67e22')
            
            # Add value labels
            for i, (train, test) in enumerate(zip(train_values, test_values)):
                ax.text(i - bar_width/2, train, f'{train:.2f}',
                       ha='center', va='bottom', fontsize=8)
                ax.text(i + bar_width/2, test, f'{test:.2f}',
                       ha='center', va='bottom', fontsize=8)
            
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models)
            if idx == 0:
                ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Local Trajectory Metrics Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'local_metrics.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(self.figures_dir / 'local_metrics.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    def plot_performance_radar(self):
        """Figure 8: Overall performance radar chart"""
        print("  🎯 Creating performance radar chart...")
        
        # Prepare data - normalize to 0-100 scale (higher is better)
        categories = ['OD\nCoverage', 'Distance\nQuality', 'Radius\nQuality',
                     'Distance\nAccuracy', 'Local\nQuality']
        
        models_data = {}
        
        for model in ['distilled', 'distilled_seed44', 'vanilla']:
            # Average across train and test
            od_coverage = []
            dist_jsd = []
            rad_jsd = []
            dist_acc = []
            hausdorff = []
            
            for result in self.results:
                m, o = self._parse_model_info(result)
                if m == model:
                    matched = result['matched_od_pairs']
                    total = result['total_generated_od_pairs']
                    od_coverage.append((matched / total) * 100)
                    
                    # JSD: convert to quality score (lower is better, so invert)
                    dist_jsd.append(1 - min(result['Distance_JSD'], 1))
                    rad_jsd.append(1 - min(result['Radius_JSD'], 1))
                    
                    # Distance accuracy: closeness to 5.16 km
                    real_dist = result['Distance_real_mean']
                    gen_dist = result['Distance_gen_mean']
                    acc = 1 - abs(gen_dist - real_dist) / real_dist
                    dist_acc.append(max(0, acc) * 100)
                    
                    # Hausdorff: normalize (lower is better for absolute value,
                    # but need context of trajectory length)
                    hausdorff.append(result['Hausdorff_km'])
            
            if od_coverage:
                models_data[model] = [
                    np.mean(od_coverage),
                    np.mean(dist_jsd) * 100,
                    np.mean(rad_jsd) * 100,
                    np.mean(dist_acc),
                    (1 - np.mean(hausdorff) / 2) * 100  # Normalize Hausdorff
                ]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for model, values in models_data.items():
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=model,
                   color=COLORS.get(model, 'gray'), markersize=8)
            ax.fill(angles, values, alpha=0.15, color=COLORS.get(model, 'gray'))
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=11)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], size=9)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title('Overall Performance Comparison\n(Higher is Better)', 
                    size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'performance_radar.pdf', bbox_inches='tight', dpi=300)
        plt.savefig(self.figures_dir / 'performance_radar.png', bbox_inches='tight', dpi=300)
        plt.close()


def main():
    """Main execution"""
    print("=" * 60)
    print("HOSER Distillation: Evaluation Figure Generation")
    print("=" * 60)
    
    visualizer = EvaluationVisualizer()
    visualizer.create_all_figures()
    
    print("\n" + "=" * 60)
    print("✅ Figure generation complete!")
    print("=" * 60)
    print(f"\nFigures saved to: {visualizer.figures_dir.absolute()}")
    print("\nGenerated figures:")
    for pdf in sorted(visualizer.figures_dir.glob("*.pdf")):
        print(f"  • {pdf.name}")


if __name__ == '__main__':
    main()

