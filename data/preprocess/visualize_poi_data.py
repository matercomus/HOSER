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
            # Load larger sample to ensure geographic coverage, then filter
            poi_df = pl.scan_csv(poi_file, encoding='utf8').head(sample_size * 3).collect()
            
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
            
            # Randomly sample from filtered data to ensure geographic diversity
            if len(poi_df) > sample_size:
                poi_df = poi_df.sample(n=sample_size, seed=42)
            
            # Convert to numpy arrays for plotting
            self.poi_points = {
                'lon': poi_df.select('大地X').to_numpy().flatten(),
                'lat': poi_df.select('大地Y').to_numpy().flatten(),
                'categories': poi_df.select('类型1').to_numpy().flatten()
            }
            
            print(f"✅ Loaded {len(self.poi_points['lon'])} POI points for visualization")
            print(f"   Latitude range: {self.poi_points['lat'].min():.3f} to {self.poi_points['lat'].max():.3f}")
            
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
        """Plot POI density across zones with geographic context"""
        print("📊 Creating zone POI density plot with geographic context...")
        
        # Calculate total POIs per zone
        zone_totals = self.density_matrix.sum(axis=1)
        zones_with_pois = np.sum(zone_totals > 0)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('POI Density Analysis Across HOSER Zones', fontsize=16, fontweight='bold', y=0.95)
        
        # Plot 1: Histogram of POI counts per zone
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
        
        # Plot 2: Box plots for top 10 POI categories (with outlier filtering)
        # Get top 10 categories by total POI count
        category_totals = np.sum(self.density_matrix, axis=0)
        top_10_indices = np.argsort(category_totals)[-10:][::-1]  # Top 10, descending order
        top_10_categories = [self.english_categories[i] for i in top_10_indices]
        
        # Prepare data for box plots with outlier filtering
        box_data = []
        filtered_stats = []
        
        for cat_idx in top_10_indices:
            category_data = self.density_matrix[:, cat_idx]
            # Only include zones with POIs in this category
            non_zero_data = category_data[category_data > 0]
            
            # Filter extreme outliers (top 1% of values) to better show distribution
            if len(non_zero_data) > 0:
                threshold = np.percentile(non_zero_data, 99)
                filtered_data = non_zero_data[non_zero_data <= threshold]
                box_data.append(filtered_data)
                filtered_stats.append({
                    'original_count': len(non_zero_data),
                    'filtered_count': len(filtered_data),
                    'max_original': np.max(non_zero_data),
                    'max_filtered': np.max(filtered_data) if len(filtered_data) > 0 else 0
                })
            else:
                box_data.append(non_zero_data)
                filtered_stats.append({
                    'original_count': 0,
                    'filtered_count': 0,
                    'max_original': 0,
                    'max_filtered': 0
                })
        
        # Create box plots
        box_plot = ax2.boxplot(box_data, patch_artist=True, labels=top_10_categories)
        
        # Color the boxes with different colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(box_data)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_xlabel('POI Category', fontsize=12)
        ax2.set_ylabel('POI Count per Zone (number of POIs)', fontsize=12)
        ax2.set_title('POI Distribution by Category (Top 10 Categories, Outliers Filtered)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style='plain', axis='y')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        
        # Add summary statistics
        total_pois = np.sum(category_totals[top_10_indices])
        total_filtered = sum(len(data) for data in box_data)
        stats_text = f'Top 10 Categories (99th percentile):\nTotal POIs: {total_pois:,.0f}\nZones shown: {total_filtered:,}'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontsize=9)
        
        # Plot 3: Geographic distribution of zones by POI count
        if self.hoser_zones and self.beijing_boundary is not None:
            # Prepare zone data
            zone_data = []
            for zone_id, zone_info in self.hoser_zones.items():
                center = zone_info['center']
                zone_data.append({
                    'zone_id': int(zone_id),
                    'lon': center['lon'],
                    'lat': center['lat'],
                    'poi_count': zone_info['poi_count']
                })
            
            # Create GeoDataFrame
            from shapely.geometry import Point
            zones_gdf = gpd.GeoDataFrame(
                zone_data,
                geometry=[Point(xy) for xy in zip([z['lon'] for z in zone_data], [z['lat'] for z in zone_data])],
                crs='EPSG:4326'
            )
            
            # Plot Beijing boundary
            self.beijing_boundary.boundary.plot(ax=ax3, color='black', linewidth=2, alpha=0.8)
            
            # Plot zones colored by POI count
            zones_gdf.plot(column='poi_count', 
                          cmap='viridis', 
                          markersize=80,
                          alpha=0.8,
                          ax=ax3,
                          legend=True,
                          legend_kwds={'shrink': 0.8, 'label': 'POI Count'})
            
            ax3.set_xlabel('Longitude (degrees)', fontsize=12)
            ax3.set_ylabel('Latitude (degrees)', fontsize=12)
            ax3.set_title('Geographic Distribution of POI Density', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Set axis limits
            bounds = self.beijing_boundary.total_bounds
            ax3.set_xlim(bounds[0] - 0.1, bounds[2] + 0.1)
            ax3.set_ylim(bounds[1] - 0.1, bounds[3] + 0.1)
        else:
            ax3.text(0.5, 0.5, 'Geographic data not available', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Geographic Distribution of POI Density', fontsize=14, fontweight='bold')
        
        # Plot 4: POI count vs zone area (if available)
        if self.hoser_zones:
            zone_areas = []
            zone_counts = []
            for zone_id, zone_info in self.hoser_zones.items():
                if 'area' in zone_info:
                    zone_areas.append(zone_info['area'])
                    zone_counts.append(zone_info['poi_count'])
            
            if zone_areas:
                ax4.scatter(zone_areas, zone_counts, alpha=0.6, color='purple', s=50)
                ax4.set_xlabel('Zone Area (km²)', fontsize=12)
                ax4.set_ylabel('POI Count', fontsize=12)
                ax4.set_title('POI Count vs Zone Area', fontsize=14, fontweight='bold')
                ax4.grid(True, alpha=0.3)
                ax4.ticklabel_format(style='plain', axis='both')
                
                # Add correlation coefficient
                correlation = np.corrcoef(zone_areas, zone_counts)[0, 1]
                ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                        transform=ax4.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontsize=10)
            else:
                ax4.text(0.5, 0.5, 'Zone area data not available', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('POI Count vs Zone Area', fontsize=14, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Zone data not available', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('POI Count vs Zone Area', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'zone_poi_density.pdf', dpi=300, bbox_inches='tight')
        print(f"✅ Zone POI density plot with geographic context saved to {self.data_dir / 'zone_poi_density.pdf'}")
        plt.close()
        
    def plot_poi_density_heatmap(self):
        """Create geographic POI density heatmap using GeoPandas and spatial interpolation"""
        print("📊 Creating geographic POI density heatmap...")
        
        if not self.hoser_zones or self.beijing_boundary is None:
            print("❌ Missing required data for geographic density plotting")
            return
        
        # Prepare zone data with POI counts
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
            print("❌ No zone data available")
            return
        
        # Create GeoDataFrame
        from shapely.geometry import Point
        zones_gdf = gpd.GeoDataFrame(
            zone_data,
            geometry=[Point(xy) for xy in zip([z['lon'] for z in zone_data], [z['lat'] for z in zone_data])],
            crs='EPSG:4326'
        )
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('Geographic POI Density Analysis', fontsize=16, fontweight='bold', y=0.95)
        
        # Plot 1: Geographic POI Density Surface
        self.beijing_boundary.boundary.plot(ax=ax1, color='black', linewidth=2, alpha=0.8)
        
        # Create a smooth density surface using interpolation
        from scipy.interpolate import griddata
        
        # Get Beijing bounds
        bounds = self.beijing_boundary.total_bounds
        lon_min, lon_max = bounds[0], bounds[2]
        lat_min, lat_max = bounds[1], bounds[3]
        
        # Create a regular grid for interpolation
        grid_resolution = 100
        lon_grid = np.linspace(lon_min, lon_max, grid_resolution)
        lat_grid = np.linspace(lat_min, lat_max, grid_resolution)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # Prepare data for interpolation
        points = np.column_stack([zones_gdf.geometry.x, zones_gdf.geometry.y])
        values = zones_gdf['poi_count'].values
        
        # Interpolate POI density across the grid
        density_surface = griddata(points, values, (lon_mesh, lat_mesh), method='cubic', fill_value=0)
        
        # Create a mask for areas outside Beijing boundary
        from shapely.geometry import Point as ShapelyPoint
        beijing_mask = np.zeros_like(density_surface, dtype=bool)
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                point = ShapelyPoint(lon_mesh[i, j], lat_mesh[i, j])
                beijing_mask[i, j] = self.beijing_boundary.geometry.apply(
                    lambda geom: geom.contains(point)
                ).any()
        
        # Apply mask to density surface
        density_surface[~beijing_mask] = np.nan
        
        # Plot the interpolated density surface
        im1 = ax1.imshow(density_surface, 
                        extent=[lon_min, lon_max, lat_min, lat_max], 
                        origin='lower', 
                        cmap='YlOrRd', 
                        alpha=0.8, 
                        aspect='auto')
        
        ax1.set_xlabel('Longitude (degrees)', fontsize=12)
        ax1.set_ylabel('Latitude (degrees)', fontsize=12)
        ax1.set_title('POI Density Surface (Interpolated)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('POI Count', fontsize=12)
        cbar1.ax.ticklabel_format(style='plain')
        
        # Plot 2: Zone Points with POI Count
        self.beijing_boundary.boundary.plot(ax=ax2, color='black', linewidth=2, alpha=0.8)
        
        # Plot zones colored by POI count
        zones_gdf.plot(column='poi_count', 
                      cmap='YlOrRd', 
                      markersize=100,
                      alpha=0.8,
                      ax=ax2,
                      legend=True,
                      legend_kwds={'shrink': 0.8, 'label': 'POI Count'})
        
        ax2.set_xlabel('Longitude (degrees)', fontsize=12)
        ax2.set_ylabel('Latitude (degrees)', fontsize=12)
        ax2.set_title('HOSER Zones by POI Count', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Set consistent axis limits
        for ax in [ax1, ax2]:
            ax.set_xlim(lon_min - 0.1, lon_max + 0.1)
            ax.set_ylim(lat_min - 0.1, lat_max + 0.1)
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'poi_density_heatmap.pdf', dpi=300, bbox_inches='tight')
        print(f"✅ Geographic POI density heatmap saved to {self.data_dir / 'poi_density_heatmap.pdf'}")
        plt.close()
        
    def plot_zone_geographic_distribution(self):
        """Create geographic distribution plots using GeoPandas best practices"""
        print("📊 Creating geographic distribution plots with GeoPandas...")
        
        if not self.hoser_zones or self.beijing_boundary is None:
            print("❌ Missing required data for geographic plotting")
            return
        
        # Create 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('Spatial Distribution Analysis: POI Data and HOSER Zones in Beijing', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # Prepare HOSER zones GeoDataFrame
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
            print("❌ No zone data available")
            return
        
        # Convert to GeoDataFrame
        from shapely.geometry import Point
        zones_gdf = gpd.GeoDataFrame(
            zone_data,
            geometry=[Point(xy) for xy in zip([z['lon'] for z in zone_data], [z['lat'] for z in zone_data])],
            crs='EPSG:4326'
        )
        
        # Plot 1: POI Count by HOSER Zones
        self.beijing_boundary.boundary.plot(ax=ax1, color='black', linewidth=2, alpha=0.8)
        
        # Plot zones colored by POI count with proper scaling
        zones_gdf.plot(column='poi_count', 
                      cmap='viridis', 
                      markersize=100,
                      alpha=0.8,
                      ax=ax1,
                      legend=True,
                      legend_kwds={'shrink': 0.8, 'label': 'POI Count'})
        
        ax1.set_xlabel('Longitude (degrees)', fontsize=12)
        ax1.set_ylabel('Latitude (degrees)', fontsize=12)
        ax1.set_title('POI Count by HOSER Zones', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Set consistent axis limits
        bounds = self.beijing_boundary.total_bounds
        ax1.set_xlim(bounds[0] - 0.1, bounds[2] + 0.1)
        ax1.set_ylim(bounds[1] - 0.1, bounds[3] + 0.1)
        
        # Plot 2: POI Points Distribution
        if self.poi_points is not None and len(self.poi_points['lon']) > 0:
            # Sample POI points for better visualization
            n_points = min(len(self.poi_points['lon']), 5000)
            if n_points < len(self.poi_points['lon']):
                # Random sampling for better geographic coverage
                import random
                random.seed(42)
                indices = random.sample(range(len(self.poi_points['lon'])), n_points)
                lon_sample = [self.poi_points['lon'][i] for i in indices]
                lat_sample = [self.poi_points['lat'][i] for i in indices]
            else:
                lon_sample = self.poi_points['lon']
                lat_sample = self.poi_points['lat']
            
            # Create POI points GeoDataFrame
            poi_gdf = gpd.GeoDataFrame(
                {'lon': lon_sample, 'lat': lat_sample},
                geometry=[Point(xy) for xy in zip(lon_sample, lat_sample)],
                crs='EPSG:4326'
            )
            
            # Clip POI points to Beijing boundary using GeoPandas
            poi_clipped = gpd.clip(poi_gdf, self.beijing_boundary)
            
            # Plot Beijing boundary
            self.beijing_boundary.boundary.plot(ax=ax2, color='black', linewidth=2, alpha=0.8)
            
            # Plot clipped POI points
            poi_clipped.plot(ax=ax2, 
                           color='red', 
                           markersize=1,
                           alpha=0.6,
                           label=f'POI Points (n={len(poi_clipped):,})')
            
            ax2.set_xlabel('Longitude (degrees)', fontsize=12)
            ax2.set_ylabel('Latitude (degrees)', fontsize=12)
            ax2.set_title(f'POI Points Distribution (n={len(poi_clipped):,})', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No POI points data available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('POI Points Distribution', fontsize=14, fontweight='bold')
        
        # Set consistent axis limits
        ax2.set_xlim(bounds[0] - 0.1, bounds[2] + 0.1)
        ax2.set_ylim(bounds[1] - 0.1, bounds[3] + 0.1)
        
        # Plot 3: HOSER Zone Coverage
        self.beijing_boundary.boundary.plot(ax=ax3, color='black', linewidth=2, alpha=0.8)
        
        # Plot all HOSER zones
        zones_gdf.plot(ax=ax3, 
                      color='blue', 
                      markersize=50,
                      alpha=0.7,
                      label=f'HOSER Zones (n={len(zones_gdf)})')
        
        ax3.set_xlabel('Longitude (degrees)', fontsize=12)
        ax3.set_ylabel('Latitude (degrees)', fontsize=12)
        ax3.set_title(f'HOSER Zone Coverage (n={len(zones_gdf)})', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Set consistent axis limits
        ax3.set_xlim(bounds[0] - 0.1, bounds[2] + 0.1)
        ax3.set_ylim(bounds[1] - 0.1, bounds[3] + 0.1)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = self.data_dir / 'zone_geographic_distribution.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Geographic distribution plot saved to {output_path}")
        
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
