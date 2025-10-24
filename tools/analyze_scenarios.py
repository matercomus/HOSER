#!/usr/bin/env python3
"""
Standalone post-processing script for scenario-based trajectory analysis.

Runs AFTER python_pipeline.py completes. Takes an evaluation directory and automatically
discovers all necessary files (generated trajectories, real data, geo data, dataset config).
Saves results in a 'scenarios/' subdirectory within the evaluation directory.

Usage:
    # After running python_pipeline.py
    uv run python tools/analyze_scenarios.py \
        --eval-dir hoser-distill-optuna-6/evaluation_20241024_123456 \
        --config config/scenarios/porto.yaml

    # Analyze specific models only
    uv run python tools/analyze_scenarios.py \
        --eval-dir evaluation_xyz \
        --config config/scenarios/beijing.yaml \
        --models vanilla,distilled

Features:
- Auto-discovers generated trajectories, real data, geo files from eval dir
- Categorizes trajectories by temporal/spatial/functional scenarios
- Computes metrics per scenario (individual + combinations)
- Hierarchical breakdown (e.g., airport → peak vs off-peak)
- Statistical significance tests
- Visualization plots
- Results saved to eval_dir/scenarios/
"""

import argparse
from pathlib import Path
import json
import ast
import polars as pl
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import yaml
from datetime import datetime
import sys
import logging

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import existing evaluation metrics (reuse, don't rewrite)
from evaluation import GlobalMetrics, LocalMetrics

# Additional imports for geo processing
from shapely.geometry import LineString

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ScenarioConfig:
    """Parsed scenario configuration"""
    dataset: str
    temporal: dict
    spatial: dict
    trip_types: dict
    analysis: dict
    
    @classmethod
    def from_yaml(cls, config_path: Path):
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


class ScenarioCategorizer:
    """Categorize OD pairs into scenarios"""
    
    def __init__(self, config: ScenarioConfig, geo_df: pl.DataFrame):
        self.config = config
        self.geo_df = geo_df
        self._prepare_spatial_lookups()
    
    def _prepare_spatial_lookups(self):
        """Pre-compute spatial data for fast lookups"""
        # Build road_id → (lat, lon) mapping
        self.road_coords = {}
        
        # Determine ID column name
        id_col = 'road_id' if 'road_id' in self.geo_df.columns else 'geo_id'
        
        for row in self.geo_df.iter_rows(named=True):
            # Extract first coordinate from the road segment
            coords = json.loads(row['coordinates']) if isinstance(row['coordinates'], str) else row['coordinates']
            if coords and len(coords) > 0:
                # coordinates are in [lon, lat] format
                self.road_coords[row[id_col]] = (coords[0][1], coords[0][0])  # lat, lon
        
        # Pre-compute airport road IDs (if using coordinates method)
        if self.config.spatial['airport']['enabled']:
            self.airport_road_ids = self._find_airport_roads()
    
    def _find_airport_roads(self) -> set:
        """Find all road IDs within airport radius"""
        try:
            from geopy.distance import geodesic
        except ImportError:
            logger.warning("geopy not installed, using simple distance calculation")
            geodesic = None
        
        airport_roads = set()
        method = self.config.spatial['airport']['method']
        
        if method in ['coordinates', 'hybrid']:
            for airport in self.config.spatial['airport']['airports']:
                airport_center = (airport['lat'], airport['lon'])
                radius_km = airport['radius_km']
                
                for road_id, coords in self.road_coords.items():
                    if geodesic:
                        distance_km = geodesic(airport_center, coords).km
                    else:
                        # Simple Euclidean approximation
                        lat_diff = airport_center[0] - coords[0]
                        lon_diff = airport_center[1] - coords[1]
                        distance_km = np.sqrt(lat_diff**2 + lon_diff**2) * 111  # rough km conversion
                    
                    if distance_km <= radius_km:
                        airport_roads.add(road_id)
        
        # Add manual fallback IDs
        if method == 'hybrid' and 'fallback_road_ids' in self.config.spatial['airport']:
            airport_roads.update(self.config.spatial['airport']['fallback_road_ids'])
        
        return airport_roads
    
    def categorize_od_pair(self, origin_road_id: int, dest_road_id: int, 
                          timestamp: datetime) -> Dict:
        """
        Categorize a single OD pair into scenarios.
        
        Returns:
            {
                'is_peak': bool,
                'is_weekend': bool,
                'origin_in_center': bool,
                'dest_in_center': bool,
                'is_airport_trip': bool,
                'trip_type': str,
                'scenario_tags': list[str]
            }
        """
        scenarios = {}
        
        # Temporal categorization
        if self.config.temporal['weekend']['enabled']:
            scenarios['is_weekend'] = timestamp.weekday() >= 5
        
        if self.config.temporal['peak_hours']['enabled']:
            scenarios['is_peak'] = self._is_peak_hour(timestamp)
        
        # Spatial categorization
        origin_coords = self.road_coords.get(origin_road_id)
        dest_coords = self.road_coords.get(dest_road_id)
        
        if self.config.spatial['city_center']['enabled'] and origin_coords and dest_coords:
            scenarios['origin_in_center'] = self._is_city_center(origin_coords)
            scenarios['dest_in_center'] = self._is_city_center(dest_coords)
        
        if self.config.spatial['airport']['enabled']:
            scenarios['origin_is_airport'] = origin_road_id in self.airport_road_ids
            scenarios['dest_is_airport'] = dest_road_id in self.airport_road_ids
            scenarios['is_airport_trip'] = scenarios['origin_is_airport'] or scenarios['dest_is_airport']
        
        # Trip type classification
        if self.config.trip_types['enabled']:
            scenarios['trip_type'] = self._classify_trip_type(scenarios)
        
        # Generate tags for easy filtering
        scenarios['scenario_tags'] = self._generate_tags(scenarios)
        
        return scenarios
    
    def _is_peak_hour(self, timestamp: datetime) -> bool:
        """Check if timestamp falls in peak hours"""
        if timestamp.weekday() >= 5:  # Weekend
            return False
        
        hour = timestamp.hour
        minute = timestamp.minute
        time_minutes = hour * 60 + minute
        
        peak_config = self.config.temporal['peak_hours']
        
        # Morning peak
        morning_start = self._parse_time(peak_config['weekday_morning'][0])
        morning_end = self._parse_time(peak_config['weekday_morning'][1])
        if morning_start <= time_minutes < morning_end:
            return True
        
        # Evening peak
        evening_start = self._parse_time(peak_config['weekday_evening'][0])
        evening_end = self._parse_time(peak_config['weekday_evening'][1])
        if evening_start <= time_minutes < evening_end:
            return True
        
        return False
    
    def _parse_time(self, time_str: str) -> int:
        """Convert 'HH:MM' to minutes since midnight"""
        h, m = map(int, time_str.split(':'))
        return h * 60 + m
    
    def _is_city_center(self, coords: tuple) -> bool:
        """Check if coordinates are in city center"""
        try:
            from geopy.distance import geodesic
        except ImportError:
            # Simple Euclidean approximation
            center_config = self.config.spatial['city_center']
            center = (center_config['center_lat'], center_config['center_lon'])
            lat_diff = center[0] - coords[0]
            lon_diff = center[1] - coords[1]
            distance_km = np.sqrt(lat_diff**2 + lon_diff**2) * 111  # rough km conversion
            return distance_km <= center_config['radius_km']
        
        center_config = self.config.spatial['city_center']
        center = (center_config['center_lat'], center_config['center_lon'])
        distance_km = geodesic(coords, center).km
        
        return distance_km <= center_config['radius_km']
    
    def _classify_trip_type(self, scenarios: dict) -> str:
        """Classify trip type based on origin/dest"""
        if scenarios.get('dest_is_airport'):
            return 'to_airport'
        elif scenarios.get('origin_is_airport'):
            return 'from_airport'
        elif scenarios.get('dest_in_center') and not scenarios.get('origin_in_center'):
            return 'to_center'
        elif scenarios.get('origin_in_center') and not scenarios.get('dest_in_center'):
            return 'from_center'
        elif scenarios.get('origin_in_center') and scenarios.get('dest_in_center'):
            return 'within_center'
        else:
            return 'suburban'
    
    def _generate_tags(self, scenarios: dict) -> list:
        """Generate list of scenario tags for filtering"""
        tags = []
        
        if scenarios.get('is_peak') is not None:
            tags.append('peak' if scenarios['is_peak'] else 'off_peak')
        
        if scenarios.get('is_weekend') is not None:
            tags.append('weekend' if scenarios['is_weekend'] else 'weekday')
        
        if scenarios.get('is_airport_trip'):
            tags.append('airport')
        
        if scenarios.get('origin_in_center') or scenarios.get('dest_in_center'):
            tags.append('city_center')
        
        if 'trip_type' in scenarios:
            tags.append(scenarios['trip_type'])
        
        return tags


class ScenarioAnalyzer:
    """Main analyzer for computing scenario-based metrics"""
    
    def __init__(self, config: ScenarioConfig, geo_df: pl.DataFrame):
        self.config = config
        self.geo_df = geo_df
        
        # Rename geo_id to road_id for consistency
        geo_df_renamed = geo_df
        if "geo_id" in geo_df.columns and "road_id" not in geo_df.columns:
            geo_df_renamed = geo_df.rename({"geo_id": "road_id"})
        
        self.geo_df_pandas = geo_df_renamed.to_pandas()  # Convert for evaluation metrics
        self.categorizer = ScenarioCategorizer(config, geo_df)
    
    def analyze(self, generated_df: pl.DataFrame, real_df: pl.DataFrame) -> Dict:
        """
        Main analysis entry point.
        
        Returns comprehensive scenario analysis results.
        """
        logger.info("📊 Starting scenario analysis...")
        
        # Step 1: Categorize all trajectories
        logger.info("  1. Categorizing trajectories by scenarios...")
        scenario_data = self._categorize_all_trajectories(generated_df)
        
        # Step 2: Individual scenario metrics
        logger.info("  2. Computing individual scenario metrics...")
        individual_metrics = self._compute_individual_scenarios(
            scenario_data, generated_df, real_df
        )
        
        # Step 3: Combination metrics
        logger.info("  3. Computing scenario combination metrics...")
        combination_metrics = self._compute_combinations(
            scenario_data, generated_df, real_df
        )
        
        # Step 4: Hierarchical breakdown
        logger.info("  4. Computing hierarchical breakdowns...")
        hierarchical_metrics = self._compute_hierarchical(
            scenario_data, generated_df, real_df
        )
        
        # Step 5: Statistical tests
        logger.info("  5. Running statistical significance tests...")
        statistical_tests = self._run_statistical_tests(
            scenario_data, generated_df, real_df, individual_metrics
        )
        
        # Compile results
        results = {
            'overview': self._compute_overview(scenario_data),
            'individual_scenarios': individual_metrics,
            'combinations': combination_metrics,
            'hierarchical': hierarchical_metrics,
            'statistics': statistical_tests
        }
        
        logger.info("✅ Scenario analysis complete!")
        return results
    
    def _categorize_all_trajectories(self, generated_df: pl.DataFrame) -> List[Dict]:
        """Categorize all trajectories"""
        scenario_data = []
        
        for row in generated_df.iter_rows(named=True):
            origin_road_id = row['origin_road_id']
            dest_road_id = row['destination_road_id']
            timestamp = datetime.fromisoformat(row['source_origin_time'].replace('Z', '+00:00'))
            
            scenarios = self.categorizer.categorize_od_pair(
                origin_road_id, dest_road_id, timestamp
            )
            scenario_data.append(scenarios)
        
        return scenario_data
    
    def _compute_overview(self, scenario_data: List[Dict]) -> Dict:
        """Compute overview statistics"""
        scenario_counts = {}
        for scenario in scenario_data:
            for tag in scenario['scenario_tags']:
                scenario_counts[tag] = scenario_counts.get(tag, 0) + 1
        
        return {
            'total_trajectories': len(scenario_data),
            'scenario_distribution': scenario_counts
        }
    
    def _compute_individual_scenarios(self, scenario_data, generated_df, real_df) -> Dict:
        """Compute metrics for individual scenario tags"""
        results = {}
        
        all_tags = set()
        for s in scenario_data:
            all_tags.update(s['scenario_tags'])
        
        min_samples = self.config.analysis['min_samples_per_scenario']
        
        for tag in sorted(all_tags):  # Sort for consistent output
            indices = [i for i, s in enumerate(scenario_data) if tag in s['scenario_tags']]
            
            if len(indices) < min_samples:
                logger.info(f"    Skipping '{tag}' - only {len(indices)} samples (min: {min_samples})")
                continue
            
            logger.info(f"    Computing metrics for '{tag}' ({len(indices)} trajectories)...")
            
            # Compute metrics for this subset
            metrics = self._compute_subset_metrics(indices, generated_df, real_df)
            
            results[tag] = {
                'count': len(indices),
                'percentage': (len(indices) / len(scenario_data)) * 100,
                'metrics': metrics
            }
        
        return results
    
    def _compute_combinations(self, scenario_data, generated_df, real_df) -> Dict:
        """Compute metrics for scenario combinations"""
        if not self.config.analysis['reporting']['combinations']:
            return {}
        
        results = {}
        combination_groups = {}
        
        # Group by unique combinations
        for i, scenario in enumerate(scenario_data):
            combo = tuple(sorted(scenario['scenario_tags']))
            if combo not in combination_groups:
                combination_groups[combo] = []
            combination_groups[combo].append(i)
        
        min_samples = self.config.analysis['min_samples_per_scenario']
        
        # Process top combinations (limit to avoid excessive computation)
        sorted_combos = sorted(combination_groups.items(), key=lambda x: len(x[1]), reverse=True)[:20]
        
        for combo, indices in sorted_combos:
            if len(indices) < min_samples:
                continue
            
            combo_name = '+'.join(combo)
            logger.info(f"    Computing metrics for '{combo_name}' ({len(indices)} trajectories)...")
            
            metrics = self._compute_subset_metrics(indices, generated_df, real_df)
            
            results[combo_name] = {
                'count': len(indices),
                'percentage': (len(indices) / len(scenario_data)) * 100,
                'metrics': metrics
            }
        
        return results
    
    def _compute_hierarchical(self, scenario_data, generated_df, real_df) -> Dict:
        """Compute hierarchical breakdowns (e.g., airport → peak vs off-peak)"""
        if not self.config.analysis['reporting']['hierarchical']:
            return {}
        
        results = {}
        min_samples = self.config.analysis['min_samples_per_scenario']
        
        # Define hierarchies to analyze
        hierarchies = [
            ('airport', ['peak', 'off_peak']),
            ('city_center', ['peak', 'off_peak']),
            ('weekday', ['to_airport', 'from_airport', 'to_center', 'from_center']),
        ]
        
        for parent_tag, child_tags in hierarchies:
            # Find all trajectories with parent tag
            parent_indices = [i for i, s in enumerate(scenario_data) if parent_tag in s['scenario_tags']]
            
            if len(parent_indices) < min_samples:
                continue
            
            breakdown = {}
            for child_tag in child_tags:
                # Find subset that has both parent AND child
                subset_indices = [i for i in parent_indices 
                                if child_tag in scenario_data[i]['scenario_tags']]
                
                if len(subset_indices) >= min_samples:
                    logger.info(f"    Computing hierarchical: {parent_tag} → {child_tag} ({len(subset_indices)} trajectories)...")
                    
                    metrics = self._compute_subset_metrics(subset_indices, generated_df, real_df)
                    breakdown[child_tag] = {
                        'count': len(subset_indices),
                        'percentage': (len(subset_indices) / len(parent_indices)) * 100,
                        'metrics': metrics
                    }
            
            if breakdown:
                results[parent_tag] = breakdown
        
        return results
    
    def _compute_subset_metrics(self, indices: List[int], generated_df, real_df) -> Dict:
        """Compute evaluation metrics for a subset of trajectories"""
        # Extract subset DataFrames
        generated_subset_df = generated_df[indices]
        
        # Map generated trajectories to their source indices in real data
        source_indices = generated_subset_df['source_index'].to_list()
        real_subset_df = real_df[source_indices]
        
        # Parse trajectories to expected format: [(road_id, timestamp), ...]
        generated_subset = []
        real_subset = []
        
        for row in generated_subset_df.iter_rows(named=True):
            traj = self._parse_generated_trajectory_to_tuples(row)
            if traj:
                generated_subset.append(traj)
        
        for row in real_subset_df.iter_rows(named=True):
            traj = self._parse_real_trajectory_to_tuples(row)
            if traj:
                real_subset.append(traj)
        
        # Compute metrics (reuse existing evaluation logic)
        try:
            # Pass pandas DataFrame to metrics
            global_metrics = GlobalMetrics(real_subset, generated_subset, self.geo_df_pandas).evaluate()
            local_metrics = LocalMetrics(real_subset, generated_subset, self.geo_df_pandas).evaluate()
            
            return {**global_metrics, **local_metrics}
        except Exception as e:
            logger.warning(f"Error computing metrics: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': str(e),
                'Distance_JSD': float('nan'),
                'Duration_JSD': float('nan'),
                'Hausdorff_mean': float('nan')
            }
    
    def _parse_generated_trajectory(self, row: Dict) -> Dict:
        """Parse generated trajectory from dataframe row"""
        # Generated trajectory format
        return {
            'rid_list': json.loads(row['gene_trace_road_id']),
            'time_list': json.loads(row['gene_trace_datetime'])
        }
    
    def _parse_real_trajectory(self, row: Dict) -> Dict:
        """Parse real trajectory from dataframe row"""
        # Real trajectory format
        return {
            'rid_list': [int(x) for x in row['rid_list'].split(',')],
            'time_list': row['time_list'].split(',')
        }
    
    def _parse_generated_trajectory_to_tuples(self, row: Dict) -> List[Tuple]:
        """Parse generated trajectory to expected format: [(road_id, timestamp), ...]"""
        try:
            # Parse road IDs and timestamps
            road_ids = json.loads(row['gene_trace_road_id'])
            timestamps_str = json.loads(row['gene_trace_datetime'])
            
            # Convert timestamps to datetime objects
            timestamps = []
            for ts in timestamps_str:
                # Handle different timestamp formats
                ts = ts.strip('"')
                try:
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                except:
                    dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
                timestamps.append(dt)
            
            # Zip into tuples
            if len(road_ids) == len(timestamps):
                return list(zip(road_ids, timestamps))
            else:
                logger.warning(f"Length mismatch: {len(road_ids)} roads, {len(timestamps)} timestamps")
                return None
        except Exception as e:
            logger.warning(f"Error parsing generated trajectory: {e}")
            return None
    
    def _parse_real_trajectory_to_tuples(self, row: Dict) -> List[Tuple]:
        """Parse real trajectory to expected format: [(road_id, timestamp), ...]"""
        try:
            # Parse road IDs
            road_ids = [int(x) for x in row['rid_list'].split(',')]
            
            # Parse timestamps
            timestamps_str = row['time_list'].split(',')
            timestamps = []
            for ts in timestamps_str:
                try:
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                except:
                    dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
                timestamps.append(dt)
            
            # Zip into tuples
            if len(road_ids) == len(timestamps):
                return list(zip(road_ids, timestamps))
            else:
                logger.warning(f"Length mismatch: {len(road_ids)} roads, {len(timestamps)} timestamps")
                return None
        except Exception as e:
            logger.warning(f"Error parsing real trajectory: {e}")
            return None
    
    def _run_statistical_tests(self, scenario_data, generated_df, real_df, 
                               individual_metrics) -> Dict:
        """Run statistical significance tests"""
        if not self.config.analysis['statistics']['enabled']:
            return {}
        
        results = {
            'significance_tests': {},
            'effect_sizes': {}
        }
        
        # Compare key metrics across major scenarios
        metric_keys = ['Distance_JSD', 'Duration_JSD', 'Hausdorff_mean']
        
        comparisons = [
            ('peak', 'off_peak'),
            ('weekday', 'weekend'),
            ('airport', 'city_center'),
        ]
        
        for tag1, tag2 in comparisons:
            if tag1 in individual_metrics and tag2 in individual_metrics:
                for metric_key in metric_keys:
                    if metric_key in individual_metrics[tag1]['metrics'] and \
                       metric_key in individual_metrics[tag2]['metrics']:
                        metric1 = individual_metrics[tag1]['metrics'][metric_key]
                        metric2 = individual_metrics[tag2]['metrics'][metric_key]
                        
                        # Simple comparison (in real implementation, need distributions)
                        # This is simplified - would need per-trajectory metrics
                        results['significance_tests'][f'{tag1}_vs_{tag2}_{metric_key}'] = {
                            'metric': metric_key,
                            'group1': tag1,
                            'group1_mean': metric1,
                            'group2': tag2,
                            'group2_mean': metric2,
                            'difference': abs(metric1 - metric2),
                            'relative_difference': abs(metric1 - metric2) / ((metric1 + metric2) / 2) * 100,
                            'note': 'Full statistical test requires per-trajectory distributions'
                        }
        
        return results


def create_visualizations(results: Dict, output_dir: Path):
    """Create visualization plots"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        logger.warning("matplotlib/seaborn not installed, skipping visualizations")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Scenario distribution
    plt.figure(figsize=(12, 6))
    dist = results['overview']['scenario_distribution']
    sorted_items = sorted(dist.items(), key=lambda x: x[1], reverse=True)
    tags = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]
    
    bars = plt.bar(tags, counts, color='steelblue')
    plt.title('Trajectory Distribution Across Scenarios', fontsize=16)
    plt.xlabel('Scenario', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scenario_distribution.png', dpi=150)
    plt.close()
    
    # Plot 2: Metric comparison across scenarios
    if results['individual_scenarios']:
        # Prepare data for heatmap
        scenarios = list(results['individual_scenarios'].keys())
        metrics = ['Distance_JSD', 'Duration_JSD', 'Hausdorff_mean']
        
        data = []
        for scenario in scenarios:
            row = []
            for metric in metrics:
                value = results['individual_scenarios'][scenario]['metrics'].get(metric, float('nan'))
                row.append(value)
            data.append(row)
        
        # Create heatmap
        plt.figure(figsize=(10, max(6, len(scenarios) * 0.5)))
        
        # Normalize each metric separately for better visualization
        data_array = np.array(data)
        data_normalized = np.zeros_like(data_array)
        for j in range(data_array.shape[1]):
            col = data_array[:, j]
            if not np.all(np.isnan(col)):
                col_min, col_max = np.nanmin(col), np.nanmax(col)
                if col_max > col_min:
                    data_normalized[:, j] = (col - col_min) / (col_max - col_min)
                else:
                    data_normalized[:, j] = 0.5
        
        sns.heatmap(data_normalized, 
                    xticklabels=metrics,
                    yticklabels=scenarios,
                    cmap='RdYlGn_r',
                    annot=data,
                    fmt='.3f',
                    cbar_kws={'label': 'Normalized Score'})
        
        plt.title('Evaluation Metrics by Scenario', fontsize=16)
        plt.xlabel('Metric', fontsize=12)
        plt.ylabel('Scenario', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_dir / 'metric_comparison.png', dpi=150)
        plt.close()
    
    # Plot 3: Hierarchical breakdown visualization
    if results['hierarchical']:
        for parent_scenario, breakdown in results['hierarchical'].items():
            if not breakdown:
                continue
            
            plt.figure(figsize=(8, 6))
            
            # Prepare data
            child_scenarios = list(breakdown.keys())
            counts = [breakdown[child]['count'] for child in child_scenarios]
            percentages = [breakdown[child]['percentage'] for child in child_scenarios]
            
            # Create pie chart
            colors = plt.cm.Set3(range(len(child_scenarios)))
            plt.pie(counts, labels=[f"{child}\n({pct:.1f}%)" for child, pct in zip(child_scenarios, percentages)],
                   colors=colors, autopct='%1.0f', startangle=90)
            
            plt.title(f'Breakdown of {parent_scenario.upper()} Trajectories', fontsize=16)
            plt.tight_layout()
            plt.savefig(output_dir / f'hierarchical_{parent_scenario}.png', dpi=150)
            plt.close()
    
    logger.info(f"📊 Visualizations saved to {output_dir}")


def run_scenario_analysis(generated_file: Path, dataset: str, od_source: str,
                         config_path: Path, output_dir: Path):
    """
    Main entry point for scenario analysis (called from python_pipeline.py)
    
    Args:
        generated_file: Path to generated trajectories CSV
        dataset: Dataset name (Beijing, porto_hoser)
        od_source: OD source (train, test)
        config_path: Path to scenarios config YAML
        output_dir: Output directory for results
    """
    # Load data
    logger.info(f"📂 Loading data from {generated_file.name}...")
    generated_df = pl.read_csv(generated_file)
    
    # Load real data
    # Try multiple paths to find data directory
    possible_data_dirs = [
        Path(f"data/{dataset}"),  # From project root
        Path(f"../data/{dataset}"),  # From eval directory
        generated_file.parent.parent.parent.parent / "data" / dataset,  # Relative to gene file
        Path("data") / dataset  # Direct path
    ]
    
    data_dir = None
    for possible_dir in possible_data_dirs:
        if possible_dir.exists():
            data_dir = possible_dir
            break
    
    if data_dir is None:
        raise FileNotFoundError(f"Could not find data directory for {dataset}")
    
    real_file = data_dir / f"{od_source}.csv"
    logger.info(f"📂 Loading real data from {real_file}...")
    real_df = pl.read_csv(real_file)
    
    # Load geo data
    geo_file = data_dir / "roadmap.geo"
    logger.info(f"📂 Loading geo data from {geo_file}...")
    # Handle complex geo file format with proper schema
    # Read first to check columns
    sample_df = pl.read_csv(geo_file, n_rows=100)
    
    # Build schema overrides based on dataset
    schema_overrides = {
        "coordinates": pl.Utf8  # Always JSON string
    }
    
    # Check for problematic columns that might contain lists
    for col in ["lanes", "oneway", "name"]:
        if col in sample_df.columns:
            try:
                # Try to read normally first
                sample_df.select(pl.col(col)).head()
            except:
                # If error, read as string
                schema_overrides[col] = pl.Utf8
    
    # For Beijing, oneway can be "[False, True]" etc
    if "oneway" in sample_df.columns and dataset.lower() == "beijing":
        schema_overrides["oneway"] = pl.Utf8
    
    # Read with proper schema
    geo_df = pl.read_csv(
        geo_file,
        infer_schema_length=10000,
        schema_overrides=schema_overrides
    )
    
    # Pre-process geo data (mimic load_road_network from evaluation.py)
    # Rename geo_id to road_id early
    if "geo_id" in geo_df.columns and "road_id" not in geo_df.columns:
        geo_df = geo_df.rename({"geo_id": "road_id"})
    
    # Calculate road center GPS coordinates
    logger.info("📐 Calculating road centroids...")
    road_center_gps = []
    for row in geo_df.iter_rows(named=True):
        try:
            coordinates = json.loads(row["coordinates"])
            road_line = LineString(coordinates=coordinates)
            center_coord = road_line.centroid
            # Store as (lat, lon) for consistency with haversine
            road_center_gps.append((center_coord.y, center_coord.x))
        except:
            road_center_gps.append((None, None))
    
    # Add center_gps column
    geo_df = geo_df.with_columns(pl.Series("center_gps", road_center_gps))
    geo_df = geo_df.filter(pl.col("center_gps").is_not_null())
    
    # Load config
    config = ScenarioConfig.from_yaml(config_path)
    
    # Run analysis
    analyzer = ScenarioAnalyzer(config, geo_df)
    results = analyzer.analyze(generated_df, real_df)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'scenario_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Visualizations can be created using create_analysis_figures.py ScenarioVisualizer
    # For backward compatibility, keep visualization code here
    create_visualizations(results, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("SCENARIO ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nDataset: {dataset}")
    print(f"OD Source: {od_source}")
    print(f"Model: {generated_file.stem}")
    print(f"\nTotal trajectories: {results['overview']['total_trajectories']}")
    print("\nScenario distribution:")
    for tag, count in sorted(results['overview']['scenario_distribution'].items(), 
                            key=lambda x: x[1], reverse=True)[:10]:
        pct = (count / results['overview']['total_trajectories']) * 100
        print(f"  {tag:<20} {count:>5} ({pct:>5.1f}%)")
    
    if results['statistics'] and results['statistics']['significance_tests']:
        print("\nKey Metric Differences:")
        for test_name, test_result in results['statistics']['significance_tests'].items():
            if 'Distance_JSD' in test_name:  # Show only distance metric for brevity
                print(f"  {test_result['group1']} vs {test_result['group2']}: "
                      f"{test_result['relative_difference']:.1f}% difference")
    
    logger.info(f"💾 Results saved to {output_dir}/")
    return results


def main():
    """Standalone CLI for scenario analysis"""
    parser = argparse.ArgumentParser(
        description='Post-processing scenario analysis for trajectory generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze evaluation directory
  uv run python tools/analyze_scenarios.py \\
      --eval-dir hoser-distill-optuna-6-evaluation-abc123 \\
      --config config/scenarios_beijing.yaml

  # Analyze specific file
  uv run python tools/analyze_scenarios.py \\
      --generated gene/Beijing/seed42/hoser_vanilla_testod_gene.csv \\
      --real data/Beijing/test.csv \\
      --geo data/Beijing/roadmap.geo \\
      --config config/scenarios_beijing.yaml \\
      --output scenarios/test/vanilla/
"""
    )
    
    # Mutually exclusive: either eval-dir or individual files
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--eval-dir', type=str, help='Evaluation directory to analyze')
    group.add_argument('--generated', type=str, help='Generated trajectories CSV')
    
    # For individual file mode
    parser.add_argument('--real', type=str, help='Real trajectories CSV (required with --generated)')
    parser.add_argument('--geo', type=str, help='Geo data file (required with --generated)')
    parser.add_argument('--output', type=str, help='Output directory (required with --generated)')
    
    # Common arguments
    parser.add_argument('--config', type=str, required=True, help='Scenario config YAML')
    parser.add_argument('--models', type=str, help='Comma-separated list of models to analyze (for --eval-dir)')
    parser.add_argument('--od-sources', type=str, default='train,test', help='OD sources to analyze')
    
    args = parser.parse_args()
    
    if args.eval_dir:
        # Evaluation directory mode
        eval_dir = Path(args.eval_dir)
        if not eval_dir.exists():
            logger.error(f"Evaluation directory not found: {eval_dir}")
            sys.exit(1)
        
        # Auto-detect dataset from eval dir structure
        gene_dir = eval_dir / 'gene'
        if not gene_dir.exists():
            logger.error(f"No gene/ directory found in {eval_dir}")
            sys.exit(1)
        
        # Find dataset directory
        dataset_dirs = [d for d in gene_dir.iterdir() if d.is_dir()]
        if not dataset_dirs:
            logger.error(f"No dataset directory found in {gene_dir}")
            sys.exit(1)
        
        dataset = dataset_dirs[0].name
        logger.info(f"Detected dataset: {dataset}")
        
        # Copy config to eval directory
        config_path = Path(args.config)
        config_copy = eval_dir / 'config' / config_path.name
        config_copy.parent.mkdir(exist_ok=True)
        
        import shutil
        shutil.copy(config_path, config_copy)
        logger.info(f"Copied config to {config_copy}")
        
        # Process each OD source
        od_sources = args.od_sources.split(',')
        for od_source in od_sources:
            logger.info(f"\nAnalyzing {od_source} OD...")
            
            # Find all generated files for this OD source
            seed_dir = gene_dir / dataset / 'seed42'  # Assuming seed42
            if not seed_dir.exists():
                # Try to find any seed directory
                seed_dirs = list((gene_dir / dataset).glob('seed*'))
                if seed_dirs:
                    seed_dir = seed_dirs[0]
                else:
                    logger.warning(f"No seed directory found in {gene_dir / dataset}")
                    continue
            
            # Try multiple patterns to find generated files
            patterns = [
                f'*{od_source}od*.csv',      # New pattern: *testod*.csv
                f'*_{od_source}_*.csv',       # Old pattern: vanilla_test_*.csv
                f'*{od_source}*.csv'          # Fallback: *test*.csv
            ]
            
            generated_files = []
            for pattern in patterns:
                files = list(seed_dir.glob(pattern))
                if files:
                    generated_files.extend(files)
                    break
            
            if not generated_files:
                logger.warning(f"No generated files found for {od_source} OD")
                continue
            
            # Filter by models if specified
            if args.models:
                model_filter = args.models.split(',')
                filtered_files = []
                for f in generated_files:
                    for model in model_filter:
                        if model in f.name:
                            filtered_files.append(f)
                            break
                generated_files = filtered_files
            
            # Process each file
            for gen_file in generated_files:
                # Extract model name from filename
                # Example: hoser_vanilla_testod_gene_20241024_123456.csv
                parts = gen_file.stem.split('_')
                if 'vanilla' in gen_file.name:
                    model_name = 'vanilla'
                elif 'distilled' in gen_file.name:
                    model_name = 'distilled'
                else:
                    model_name = 'unknown'
                
                output_dir = eval_dir / 'scenarios' / od_source / model_name
                
                logger.info(f"  Analyzing {model_name} model...")
                
                try:
                    run_scenario_analysis(
                        generated_file=gen_file,
                        dataset=dataset,
                        od_source=od_source,
                        config_path=config_copy,
                        output_dir=output_dir
                    )
                except Exception as e:
                    logger.error(f"Analysis failed for {model_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        logger.info(f"\n✅ All analyses complete! Results in {eval_dir / 'scenarios'}")
        
    else:
        # Individual file mode
        if not all([args.real, args.geo, args.output]):
            parser.error("--real, --geo, and --output are required when using --generated")
        
        # Extract dataset and od_source from file paths
        generated_path = Path(args.generated)
        real_path = Path(args.real)
        
        # Try to infer dataset from path
        dataset = 'unknown'
        if 'Beijing' in str(generated_path):
            dataset = 'Beijing'
        elif 'porto' in str(generated_path).lower():
            dataset = 'porto_hoser'
        
        # Try to infer od_source from filename
        od_source = 'unknown'
        if 'test' in real_path.stem:
            od_source = 'test'
        elif 'train' in real_path.stem:
            od_source = 'train'
        elif 'val' in real_path.stem:
            od_source = 'val'
        
        logger.info(f"Inferred dataset: {dataset}, od_source: {od_source}")
        
        # Load data
        logger.info("📂 Loading data...")
        generated_df = pl.read_csv(args.generated)
        real_df = pl.read_csv(args.real)
        
        # Handle geo data with dataset-specific formats
        # Read sample first
        sample_df = pl.read_csv(args.geo, n_rows=100)
        
        # Build schema overrides
        schema_overrides = {"coordinates": pl.Utf8}
        
        # Check for problematic columns
        for col in ["lanes", "oneway", "name"]:
            if col in sample_df.columns:
                try:
                    sample_df.select(pl.col(col)).head()
                except:
                    schema_overrides[col] = pl.Utf8
        
        # For Beijing, oneway can be lists
        if "oneway" in sample_df.columns and "beijing" in dataset.lower():
            schema_overrides["oneway"] = pl.Utf8
        
        # Read with proper schema
        geo_df = pl.read_csv(
            args.geo,
            infer_schema_length=10000,
            schema_overrides=schema_overrides
        )
        
        # Load config
        config = ScenarioConfig.from_yaml(Path(args.config))
        
        # Run analysis
        analyzer = ScenarioAnalyzer(config, geo_df)
        results = analyzer.analyze(generated_df, real_df)
        
        # Save results
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'scenario_analysis.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to {output_dir / 'scenario_analysis.json'}")
        
        # Create visualizations
        create_visualizations(results, output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("SCENARIO ANALYSIS SUMMARY")
        print("="*60)
        print(f"\nTotal trajectories: {results['overview']['total_trajectories']}")
        print("\nScenario distribution:")
        for tag, count in sorted(results['overview']['scenario_distribution'].items(), 
                                key=lambda x: x[1], reverse=True)[:10]:
            pct = (count / results['overview']['total_trajectories']) * 100
            print(f"  {tag:<20} {count:>5} ({pct:>5.1f}%)")


if __name__ == '__main__':
    main()
