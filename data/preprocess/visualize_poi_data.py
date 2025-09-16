#!/usr/bin/env python3
"""
POI Data Visualization Script
Visualizes the processed POI data and its mapping to HOSER zones
"""

import polars as pl
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for better plots
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.figsize'] = (12, 8)

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class POIDataVisualizer:
    """Visualize processed POI data and HOSER zone mappings"""
    
    def __init__(self, data_dir: str = "poi_output"):
        self.data_dir = Path(data_dir)
        self.poi_categories = None
        self.zone_mapping = None
        self.hoser_zones = None
        self.density_matrix = None
        
    def load_data(self):
        """Load all processed POI data"""
        print("📊 Loading processed POI data...")
        
        # Load POI categories
        with open(self.data_dir / 'poi_categories.json', 'r') as f:
            self.poi_categories = json.load(f)
        
        # Load zone mapping
        with open(self.data_dir / 'zone_mapping.json', 'r') as f:
            self.zone_mapping = json.load(f)
        
        # Load HOSER zones with POI data
        with open(self.data_dir / 'hoser_zones_with_poi.json', 'r') as f:
            self.hoser_zones = json.load(f)
        
        # Load density matrix
        self.density_matrix = np.load(self.data_dir / 'poi_density_matrix.npy')
        
        # Set up category translation mapping
        self.category_translation = {
            '住宿服务': 'Accommodation',
            '餐饮服务': 'Food & Dining',
            '购物服务': 'Shopping',
            '生活服务': 'Daily Services',
            '体育休闲服务': 'Sports & Recreation',
            '医疗保健服务': 'Healthcare',
            '汽车服务': 'Auto Services',
            '汽车销售': 'Auto Sales',
            '汽车维修': 'Auto Repair',
            '摩托车服务': 'Motorcycle Services',
            '充电站': 'Charging Stations',
            '加油站': 'Gas Stations',
            '金融保险服务': 'Financial Services',
            '公司企业': 'Companies & Business',
            '道路附属设施': 'Road Infrastructure',
            '地名地址信息': 'Address Information',
            '公共设施': 'Public Facilities',
            '政府机构及社会团体': 'Government & Organizations',
            '科教文化服务': 'Education & Culture',
            '交通设施服务': 'Transportation',
            '风景名胜': 'Tourist Attractions',
            '商务住宅': 'Business Residential',
            '通讯营业厅': 'Telecommunications',
            '事故易发地段': 'Accident-Prone Areas',
            '室内设施': 'Indoor Facilities'
        }
        
        # Create English category list
        self.english_categories = [
            self.category_translation.get(cat, cat) for cat in self.poi_categories
        ]
        
        print(f"✅ Loaded data:")
        print(f"  - {len(self.poi_categories)} POI categories")
        print(f"  - {len(self.zone_mapping)} zones")
        print(f"  - {self.density_matrix.shape[0]} zones × {self.density_matrix.shape[1]} categories")
        
    def plot_poi_category_distribution(self):
        """Plot distribution of POI categories across all zones"""
        print("📊 Creating POI category distribution plot...")
        
        # Calculate total POIs per category
        category_totals = self.density_matrix.sum(axis=0)
        
        # Sort categories by count (highest to lowest)
        sorted_indices = np.argsort(category_totals)[::-1]
        sorted_categories = [self.english_categories[i] for i in sorted_indices]
        sorted_totals = category_totals[sorted_indices]
        
        # Create single bar plot
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        
        # Bar plot (sorted highest to lowest)
        bars = ax.bar(range(len(sorted_categories)), sorted_totals, color='steelblue', alpha=0.7)
        ax.set_xlabel('POI Category', fontsize=12)
        ax.set_ylabel('Total POI Count (number of POIs)', fontsize=12)
        ax.set_title('POI Distribution by Category (Sorted by Count)', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(sorted_categories)))
        ax.set_xticklabels(sorted_categories, rotation=45, ha='right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis to avoid scientific notation
        ax.ticklabel_format(style='plain', axis='y')
        
        # Add value labels on all bars
        for bar, value in zip(bars, sorted_totals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + sorted_totals.max() * 0.01,
                    f'{int(value):,}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'poi_category_distribution.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_zone_poi_density(self):
        """Plot POI density across zones"""
        print("📊 Creating zone POI density plot...")
        
        # Calculate total POIs per zone
        zone_totals = self.density_matrix.sum(axis=1)
        zones_with_pois = np.sum(zone_totals > 0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Histogram of POI counts per zone
        n, bins, patches = ax1.hist(zone_totals[zone_totals > 0], bins=30, alpha=0.7, 
                                   edgecolor='black', color='lightblue')
        ax1.set_xlabel('POI Count per Zone (number of POIs)', fontsize=12)
        ax1.set_ylabel('Number of Zones (count)', fontsize=12)
        ax1.set_title(f'POI Distribution Across Zones\n({zones_with_pois}/{len(zone_totals)} zones have POIs)', 
                     fontsize=14, fontweight='bold')
        ax1.axvline(zone_totals.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {zone_totals.mean():.0f}')
        ax1.axvline(np.median(zone_totals[zone_totals > 0]), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(zone_totals[zone_totals > 0]):.0f}')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(style='plain', axis='both')
        
        # Box plot with better styling
        box_plot = ax2.boxplot(zone_totals[zone_totals > 0], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightgreen')
        box_plot['boxes'][0].set_alpha(0.7)
        ax2.set_ylabel('POI Count per Zone (number of POIs)', fontsize=12)
        ax2.set_title('POI Count Distribution (Zones with POIs)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style='plain', axis='y')
        
        # Add statistics text
        stats_text = f'Min: {zone_totals[zone_totals > 0].min():.0f}\nMax: {zone_totals[zone_totals > 0].max():.0f}\nStd: {zone_totals[zone_totals > 0].std():.0f}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'zone_poi_density.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_poi_density_heatmap(self):
        """Create heatmap of POI density by zone and category"""
        print("📊 Creating POI density heatmap...")
        
        # Only show top zones with significant POI counts for readability
        zone_totals = np.sum(self.density_matrix, axis=1)
        top_zones_threshold = np.percentile(zone_totals[zone_totals > 0], 75)  # Top 25% of zones with POIs
        top_zones_mask = zone_totals >= top_zones_threshold
        filtered_matrix = self.density_matrix[top_zones_mask]
        
        # Get zone indices for labeling
        zone_indices = np.where(top_zones_mask)[0]
        zone_labels = [f"Zone {i}" for i in zone_indices]
        
        # Create figure with appropriate size
        plt.figure(figsize=(16, 12))
        
        # Create heatmap with better styling - no annotations for cleaner look
        ax = sns.heatmap(filtered_matrix, 
                        xticklabels=self.english_categories,
                        yticklabels=zone_labels,
                        cmap='YlOrRd',
                        cbar_kws={'label': 'POI Count (number of POIs)'},
                        linewidths=0.1,
                        square=False,
                        fmt='.0f')
        
        plt.title(f'POI Density Heatmap - Top {len(zone_labels)} Zones by POI Count', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('POI Category', fontsize=12)
        plt.ylabel('HOSER Zone', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=9)
        
        # Format colorbar to avoid scientific notation
        cbar = ax.collections[0].colorbar
        cbar.ax.ticklabel_format(style='plain')
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'poi_density_heatmap.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_zone_geographic_distribution(self):
        """Plot geographic distribution of zones with POIs"""
        print("📊 Creating geographic distribution plot...")
        
        # Extract zone coordinates and POI counts
        zone_data = []
        for zone_id, zone_info in self.hoser_zones.items():
            if zone_info['poi_count'] > 0:
                center = zone_info['center']
                bounds = zone_info['bounds']
                zone_data.append({
                    'zone_id': int(zone_id),
                    'lon': center['lon'],
                    'lat': center['lat'],
                    'poi_count': zone_info['poi_count'],
                    'area': zone_info['area']
                })
        
        if not zone_data:
            print("⚠️  No zones with POIs found for geographic visualization")
            return
        
        # Convert to DataFrame for easier plotting
        df = pl.DataFrame(zone_data)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Scatter plot: POI count vs location
        scatter = ax1.scatter(df['lon'], df['lat'], 
                            c=df['poi_count'], 
                            s=np.clip(df['poi_count']/100, 20, 200),  # Better size scaling
                            alpha=0.7, 
                            cmap='viridis',
                            edgecolors='black',
                            linewidth=0.5)
        ax1.set_xlabel('Longitude (degrees)', fontsize=12)
        ax1.set_ylabel('Latitude (degrees)', fontsize=12)
        ax1.set_title('Geographic Distribution of POI-Rich Zones', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('POI Count (number of POIs)', fontsize=12)
        
        # POI density vs area
        ax2.scatter(df['area'], df['poi_count'], alpha=0.7, color='steelblue', s=50)
        ax2.set_xlabel('Zone Area (square degrees)', fontsize=12)
        ax2.set_ylabel('POI Count (number of POIs)', fontsize=12)
        ax2.set_title('POI Count vs Zone Area', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style='plain', axis='both')
        
        # Add trend line
        if len(df) > 1:
            z = np.polyfit(df['area'], df['poi_count'], 1)
            p = np.poly1d(z)
            ax2.plot(df['area'], p(df['area']), "r--", alpha=0.8, linewidth=2, label=f'Trend (slope: {z[0]:.2f})')
            ax2.legend()
        
        # Add correlation coefficient
        if len(df) > 1:
            corr = np.corrcoef(df['area'], df['poi_count'])[0, 1]
            ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax2.transAxes, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'zone_geographic_distribution.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_category_correlation(self):
        """Plot correlation between different POI categories"""
        print("📊 Creating POI category correlation plot...")
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(self.density_matrix.T)
        
        # Sort categories by average correlation strength (excluding diagonal)
        avg_correlations = []
        for i in range(len(correlation_matrix)):
            # Calculate average absolute correlation for this category (excluding diagonal)
            non_diagonal = np.abs(correlation_matrix[i])
            non_diagonal[i] = 0  # Remove diagonal element
            avg_correlations.append(np.mean(non_diagonal))
        
        # Sort indices by correlation strength
        sorted_indices = np.argsort(avg_correlations)[::-1]
        sorted_categories = [self.english_categories[i] for i in sorted_indices]
        sorted_correlation_matrix = correlation_matrix[np.ix_(sorted_indices, sorted_indices)]
        
        plt.figure(figsize=(14, 12))
        
        # Create heatmap with better styling
        mask = np.triu(np.ones_like(sorted_correlation_matrix, dtype=bool))
        sns.heatmap(sorted_correlation_matrix, 
                   mask=mask,
                   xticklabels=sorted_categories,
                   yticklabels=sorted_categories,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Correlation Coefficient'},
                   linewidths=0.5)
        
        plt.title('POI Category Correlation Matrix (Sorted by Correlation Strength)', fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'poi_category_correlation.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("📋 Generating summary report...")
        
        # Calculate statistics
        total_pois = self.density_matrix.sum()
        zones_with_pois = np.sum(self.density_matrix.sum(axis=1) > 0)
        total_zones = len(self.density_matrix)
        
        # Category statistics
        category_totals = self.density_matrix.sum(axis=0)
        top_categories = sorted(zip(self.poi_categories, category_totals), 
                              key=lambda x: x[1], reverse=True)
        
        # Zone statistics
        zone_totals = self.density_matrix.sum(axis=1)
        zones_with_pois_data = zone_totals[zone_totals > 0]
        
        # Create report
        report = f"""
# POI Data Processing Summary Report

## Dataset Overview
- **Total POIs**: {total_pois:,}
- **Total Zones**: {total_zones:,}
- **Zones with POIs**: {zones_with_pois:,} ({zones_with_pois/total_zones*100:.1f}%)
- **POI Categories**: {len(self.poi_categories)}

## Top POI Categories
"""
        
        for i, (category, count) in enumerate(top_categories[:10], 1):
            percentage = count / total_pois * 100
            report += f"{i:2d}. {category}: {count:,} ({percentage:.1f}%)\n"
        
        report += f"""
## Zone Statistics
- **Average POIs per zone**: {zone_totals.mean():.1f}
- **Median POIs per zone**: {np.median(zone_totals):.1f}
- **Max POIs in a zone**: {zone_totals.max():.0f}
- **Zones with 0 POIs**: {total_zones - zones_with_pois:,}

## POI Density Distribution
- **Mean density (zones with POIs)**: {zones_with_pois_data.mean():.1f}
- **Std density (zones with POIs)**: {zones_with_pois_data.std():.1f}
- **95th percentile**: {np.percentile(zones_with_pois_data, 95):.1f}

## Data Quality
- **Coverage**: {zones_with_pois/total_zones*100:.1f}% of zones have POI data
- **Density matrix shape**: {self.density_matrix.shape}
- **Categories**: {', '.join(self.poi_categories)}
"""
        
        # Save report
        with open(self.data_dir / 'poi_summary_report.md', 'w') as f:
            f.write(report)
        
        print("✅ Summary report saved to poi_summary_report.md")
        print("\n" + "="*50)
        print(report)
        
    def run_all_visualizations(self):
        """Run all visualization functions"""
        print("🎨 Starting POI data visualization...")
        
        self.load_data()
        
        # Generate all plots
        self.plot_poi_category_distribution()
        self.plot_zone_poi_density()
        self.plot_poi_density_heatmap()
        self.plot_zone_geographic_distribution()
        self.plot_category_correlation()
        
        # Generate summary report
        self.generate_summary_report()
        
        print("✅ All visualizations completed!")
        print(f"📁 Output files saved to: {self.data_dir}")

def main():
    parser = argparse.ArgumentParser(description='Visualize processed POI data')
    parser.add_argument('--data_dir', type=str, default='poi_output',
                       help='Directory containing processed POI data')
    parser.add_argument('--plots', nargs='+', 
                       choices=['categories', 'density', 'heatmap', 'geographic', 'correlation', 'all'],
                       default=['all'],
                       help='Which plots to generate')
    
    args = parser.parse_args()
    
    visualizer = POIDataVisualizer(args.data_dir)
    visualizer.load_data()
    
    if 'all' in args.plots or 'categories' in args.plots:
        visualizer.plot_poi_category_distribution()
    
    if 'all' in args.plots or 'density' in args.plots:
        visualizer.plot_zone_poi_density()
    
    if 'all' in args.plots or 'heatmap' in args.plots:
        visualizer.plot_poi_density_heatmap()
    
    if 'all' in args.plots or 'geographic' in args.plots:
        visualizer.plot_zone_geographic_distribution()
    
    if 'all' in args.plots or 'correlation' in args.plots:
        visualizer.plot_category_correlation()
    
    visualizer.generate_summary_report()

if __name__ == "__main__":
    main()
