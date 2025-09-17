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
import geopandas as gpd
from pathlib import Path
import argparse
# from typing import Dict, List, Tuple  # Unused imports removed
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
        self.beijing_boundary = None
        self.poi_points = None
        
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
        
        print("✅ Loaded data:")
        print(f"  - {len(self.poi_categories)} POI categories")
        print(f"  - {len(self.zone_mapping)} zones")
        print(f"  - {self.density_matrix.shape[0]} zones × {self.density_matrix.shape[1]} categories")
    
    def load_beijing_boundary(self, shapefile_path: str = "/home/matt/gadm41_CHN_shp/gadm41_CHN_3.shp"):
        """Load Beijing boundary from GADM shapefile"""
        print("🗺️  Loading Beijing boundary...")
        
        try:
            # Load the shapefile
            gdf = gpd.read_file(shapefile_path)
            
            # Filter for Beijing (assuming it's identified by NAME_1 or similar field)
            # Beijing might be identified as "Beijing" or "北京市" in different fields
            beijing_fields = ['NAME_1', 'NAME_2', 'NAME_3', 'VARNAME_1', 'VARNAME_2', 'VARNAME_3']
            beijing_keywords = ['Beijing', '北京市', 'beijing', 'BEIJING']
            
            beijing_mask = None
            for field in beijing_fields:
                if field in gdf.columns:
                    for keyword in beijing_keywords:
                        if beijing_mask is None:
                            beijing_mask = gdf[field].str.contains(keyword, case=False, na=False)
                        else:
                            beijing_mask |= gdf[field].str.contains(keyword, case=False, na=False)
            
            if beijing_mask is not None and beijing_mask.any():
                self.beijing_boundary = gdf[beijing_mask]
                print(f"✅ Found Beijing boundary with {len(self.beijing_boundary)} features")
            else:
                print("⚠️  Could not find Beijing in shapefile, using all features")
                self.beijing_boundary = gdf
                
        except Exception as e:
            print(f"⚠️  Error loading Beijing boundary: {e}")
            print("   Proceeding without boundary overlay")
            self.beijing_boundary = None
    
    def load_poi_points(self, poi_file: str = "/home/matt/POI/2021_combined_utf8_no_bom.csv", sample_size: int = 5000):
        """Load a sample of POI points for visualization"""
        print("📍 Loading POI points for visualization...")
        
        try:
            # Load POI data with Polars for speed
            poi_df = pl.scan_csv(poi_file, encoding='utf8').head(sample_size).collect()
            
            # Clean coordinates and filter to Beijing bounds
            beijing_bounds = {
                'min_lon': 115.4, 'max_lon': 117.5,
                'min_lat': 39.4, 'max_lat': 41.1
            }
            
            # Filter POIs within Beijing bounds
            poi_df = poi_df.filter(
                (pl.col('大地X').cast(pl.Float64).is_not_null()) &
                (pl.col('大地Y').cast(pl.Float64).is_not_null()) &
                (pl.col('大地X') >= beijing_bounds['min_lon']) &
                (pl.col('大地X') <= beijing_bounds['max_lon']) &
                (pl.col('大地Y') >= beijing_bounds['min_lat']) &
                (pl.col('大地Y') <= beijing_bounds['max_lat'])
            )
            
            # Convert to numpy arrays for plotting
            self.poi_points = {
                'lon': poi_df.select('大地X').to_numpy().flatten(),
                'lat': poi_df.select('大地Y').to_numpy().flatten(),
                'categories': poi_df.select('类型1').to_numpy().flatten()
            }
            
            print(f"✅ Loaded {len(self.poi_points['lon'])} POI points for visualization")
            
        except Exception as e:
            print(f"⚠️  Error loading POI points: {e}")
            print("   Proceeding without POI points")
            self.poi_points = None
        
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
        """Plot geographic distribution using heatmap approach with HOSER zone grid"""
        print("📊 Creating geographic distribution heatmap...")
        
        # Create 3 subplots with better spacing
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle('Spatial Distribution Analysis: POI Data and HOSER Zones in Beijing', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Common function to plot Beijing boundary
        def plot_beijing_boundary(ax):
            if self.beijing_boundary is not None:
                try:
                    self.beijing_boundary.boundary.plot(ax=ax, color='black', linewidth=2, alpha=0.8)
                    bounds = self.beijing_boundary.total_bounds
                    ax.set_xlim(bounds[0] - 0.1, bounds[2] + 0.1)
                    ax.set_ylim(bounds[1] - 0.1, bounds[3] + 0.1)
                except Exception as e:
                    print(f"⚠️  Error plotting Beijing boundary: {e}")
        
        # Create zone grid heatmap data
        def create_zone_heatmap_data():
            """Create grid data for heatmap visualization"""
            # Get all zone centers and POI counts
            zone_data = []
            for zone_id, zone_info in self.hoser_zones.items():
                center = zone_info['center']
                zone_data.append({
                    'zone_id': int(zone_id),
                    'lon': center['lon'],
                    'lat': center['lat'],
                    'poi_count': zone_info['poi_count']
                })
            
            if not zone_data:
                return None, None, None, None
            
            df = pl.DataFrame(zone_data)
            
            # Create grid for heatmap
            lon_min, lon_max = df['lon'].min(), df['lon'].max()
            lat_min, lat_max = df['lat'].min(), df['lat'].max()
            
            # Create regular grid
            lon_bins = np.linspace(lon_min, lon_max, 50)
            lat_bins = np.linspace(lat_min, lat_max, 50)
            
            # Create 2D histogram for POI counts
            poi_heatmap, lon_edges, lat_edges = np.histogram2d(
                df['lon'], df['lat'], 
                bins=[lon_bins, lat_bins], 
                weights=df['poi_count']
            )
            
            # Create binary grid for zone presence
            zone_heatmap, _, _ = np.histogram2d(
                df['lon'], df['lat'], 
                bins=[lon_bins, lat_bins]
            )
            
            return poi_heatmap, zone_heatmap, lon_edges, lat_edges
        
        # Plot 1: POI Count Heatmap
        poi_heatmap, zone_heatmap, lon_edges, lat_edges = create_zone_heatmap_data()
        
        if poi_heatmap is not None:
            plot_beijing_boundary(ax1)
            
            # Create heatmap
            im1 = ax1.imshow(poi_heatmap.T, 
                           extent=[lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]], 
                           origin='lower', cmap='viridis', alpha=0.8, aspect='auto')
            
            ax1.set_xlabel('Longitude (degrees)', fontsize=12)
            ax1.set_ylabel('Latitude (degrees)', fontsize=12)
            ax1.set_title('POI Count Heatmap', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Add colorbar with proper range
            cbar1 = plt.colorbar(im1, ax=ax1)
            cbar1.set_label('POI Count (millions)', fontsize=12)
            cbar1.ax.ticklabel_format(style='plain')
            
            # Format colorbar to show millions
            ticks = cbar1.get_ticks()
            if len(ticks) > 0:
                cbar1.set_ticks(ticks)
                cbar1.set_ticklabels([f'{t/1e6:.1f}M' for t in ticks])
        else:
            ax1.text(0.5, 0.5, 'No POI data available', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('POI Count Heatmap', fontsize=14, fontweight='bold')
        
        # Plot 2: POI Points Density Heatmap
        if self.poi_points is not None and len(self.poi_points['lon']) > 0:
            plot_beijing_boundary(ax2)
            
            # Sample POI points for better visualization
            n_points = min(len(self.poi_points['lon']), 5000)
            if n_points < len(self.poi_points['lon']):
                indices = np.random.choice(len(self.poi_points['lon']), n_points, replace=False)
                lon_sample = self.poi_points['lon'][indices]
                lat_sample = self.poi_points['lat'][indices]
            else:
                lon_sample = self.poi_points['lon']
                lat_sample = self.poi_points['lat']
            
            # Create density heatmap for POI points
            poi_density, lon_edges, lat_edges = np.histogram2d(
                lon_sample, lat_sample, 
                bins=[50, 50]
            )
            
            im2 = ax2.imshow(poi_density.T, 
                           extent=[lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]], 
                           origin='lower', cmap='plasma', alpha=0.8, aspect='auto')
            
            ax2.set_xlabel('Longitude (degrees)', fontsize=12)
            ax2.set_ylabel('Latitude (degrees)', fontsize=12)
            ax2.set_title(f'POI Points Density (n={len(lon_sample):,})', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar2 = plt.colorbar(im2, ax=ax2)
            cbar2.set_label('POI Density (count)', fontsize=12)
            cbar2.ax.ticklabel_format(style='plain')
        else:
            ax2.text(0.5, 0.5, 'No POI points loaded', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('POI Points Density', fontsize=14, fontweight='bold')
        
        # Plot 3: Zone Presence Heatmap
        if zone_heatmap is not None:
            plot_beijing_boundary(ax3)
            
            # Create binary heatmap showing zone presence
            im3 = ax3.imshow(zone_heatmap.T, 
                           extent=[lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]], 
                           origin='lower', cmap='RdYlBu_r', alpha=0.8, aspect='auto')
            
            ax3.set_xlabel('Longitude (degrees)', fontsize=12)
            ax3.set_ylabel('Latitude (degrees)', fontsize=12)
            ax3.set_title(f'HOSER Zone Coverage (n={len(self.hoser_zones)})', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar3 = plt.colorbar(im3, ax=ax3)
            cbar3.set_label('Zone Count', fontsize=12)
            cbar3.ax.ticklabel_format(style='plain')
        else:
            ax3.text(0.5, 0.5, 'No zone data available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('HOSER Zone Coverage', fontsize=14, fontweight='bold')
        
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
    visualizer.load_beijing_boundary()
    visualizer.load_poi_points()
    
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
