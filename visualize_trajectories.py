#!/usr/bin/env python3
"""
Trajectory Visualization System

Modular system for visualizing generated, train, and test trajectories
on OpenStreetMap basemaps with configurable sampling strategies.

Usage:
    uv run python visualize_trajectories.py --sample_strategy length_based
"""

import argparse
import ast
import json
import logging
import random
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

try:
    import contextily as cx
except ImportError:
    cx = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VisualizationConfig:
    """Centralized configuration for trajectory visualization"""
    # Eval directory (required)
    eval_dir: Path = None
    dataset: str = None  # Auto-detected from evaluation.yaml if not provided
    
    # Sampling strategy
    sample_strategy: str = "length_based"  # "random", "length_based", "representative", "scenario"
    samples_per_type: int = 1
    random_seed: int = 42
    max_scenarios_to_plot: int = 5  # For scenario-based sampling
    
    # Comparison modes
    generate_separate: bool = True
    generate_overlaid: bool = True
    generate_cross_model: bool = False  # Compare all models for same OD pair
    include_real_in_cross_model: bool = True  # Include real trajectories in cross-model comparison
    generate_scenario_cross_model: bool = False  # Compare models across scenarios (per-scenario OD matching)
    
    # Background visualization
    show_road_network: bool = True  # Show road network as light gray reference
    basemap_style: str = "none"  # "osm", "gaode", "cartodb", "none" - deprecated, use road network
    basemap_timeout: int = 5  # seconds - fast timeout for China network
    basemap_test_first: bool = True  # Test connectivity before fetching all tiles
    
    # Output
    dpi: int = 300
    figsize: Tuple[int, int] = (12, 10)
    
    # Visualization
    margin: float = 0.002  # Map padding in degrees (~200m)
    
    # Paths (set by __post_init__)
    roadmap_path: Path = None
    train_csv_path: Path = None
    test_csv_path: Path = None
    gene_dir: Path = None
    output_dir: Path = None
    
    def __post_init__(self):
        """Initialize paths based on eval_dir and dataset"""
        if self.eval_dir is None:
            raise ValueError("eval_dir is required")
        
        self.eval_dir = Path(self.eval_dir).resolve()
        
        # Load configuration from evaluation.yaml
        eval_config = self._load_eval_config()
        
        # Get dataset from config or use provided
        if self.dataset is None:
            self.dataset = eval_config.get('dataset', 'Beijing')
        
        # Set paths based on dataset
        if self.dataset.lower() == "porto":
            data_dir = Path("data") / "porto_hoser"
        else:
            data_dir = Path("data") / self.dataset
        
        self.roadmap_path = data_dir / "roadmap.geo"
        self.train_csv_path = data_dir / "train.csv"
        self.test_csv_path = data_dir / "test.csv"
        self.gene_dir = self.eval_dir / "gene" / self.dataset
        self.output_dir = self.eval_dir / "figures" / "trajectories"
    
    def _load_eval_config(self) -> dict:
        """Load evaluation.yaml from eval directory"""
        config_path = self.eval_dir / "config" / "evaluation.yaml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        return {}


# =============================================================================
# Road Network Loader
# =============================================================================

class RoadNetworkLoader:
    """Load and cache road network GPS coordinates"""
    
    def __init__(self, roadmap_path: Path):
        self.roadmap_path = roadmap_path
        self.road_coords: Optional[Dict[int, List[Tuple[float, float]]]] = None
        
    def load(self) -> Dict[int, List[Tuple[float, float]]]:
        """Load roadmap.geo and create road_id -> GPS coordinates mapping"""
        if self.road_coords is not None:
            return self.road_coords
            
        logger.info(f"📂 Loading road network from {self.roadmap_path}")
        
        # Read CSV with schema overrides for mixed-type columns
        df = pl.read_csv(
            self.roadmap_path,
            schema_overrides={"lanes": pl.Utf8, "oneway": pl.Utf8}
        )
        
        # Parse coordinates
        road_coords = {}
        for row in df.iter_rows(named=True):
            road_id = row['geo_id']
            coord_str = row['coordinates']
            
            # Parse coordinates: "[[lon1, lat1], [lon2, lat2], ...]"
            try:
                coords_list = ast.literal_eval(coord_str)
                # Convert to list of (lon, lat) tuples
                road_coords[road_id] = [(lon, lat) for lon, lat in coords_list]
            except (ValueError, SyntaxError) as e:
                logger.warning(f"Failed to parse coordinates for road {road_id}: {e}")
                continue
        
        self.road_coords = road_coords
        logger.info(f"✅ Loaded {len(road_coords)} roads")
        return road_coords


# =============================================================================
# Trajectory Loader
# =============================================================================

@dataclass
class Trajectory:
    """Trajectory data structure"""
    road_ids: List[int]
    coords: List[Tuple[float, float]]
    source: str  # "generated", "train", "test"
    model: str  # "distilled", "distilled_seed44", "vanilla"
    od_type: str  # "train", "test"
    length: float  # Total length in degrees
    metadata: Dict = field(default_factory=dict)


class TrajectoryLoader:
    """Load trajectories from CSV files and convert to GPS coordinates"""
    
    def __init__(self, config: VisualizationConfig, road_network: Dict[int, List[Tuple[float, float]]]):
        self.config = config
        self.road_network = road_network
        
    def load_generated_trajectories(self, csv_path: Path, model: str, od_type: str) -> List[Trajectory]:
        """Load generated trajectories from CSV"""
        logger.info(f"📂 Loading generated trajectories from {csv_path.name}")
        
        df = pl.read_csv(csv_path)
        trajectories = []
        
        for row in df.iter_rows(named=True):
            # Parse road_id sequence
            road_id_str = row['gene_trace_road_id']
            try:
                road_ids = ast.literal_eval(road_id_str)
            except (ValueError, SyntaxError):
                logger.warning(f"Failed to parse road_ids: {road_id_str[:50]}...")
                continue
            
            # Convert to GPS coordinates
            coords = self._road_ids_to_coords(road_ids)
            if not coords:
                continue
            
            # Calculate length
            length = self._calculate_length(coords)
            
            trajectories.append(Trajectory(
                road_ids=road_ids,
                coords=coords,
                source="generated",
                model=model,
                od_type=od_type,
                length=length,
                metadata={
                    'origin_road_id': row.get('origin_road_id'),
                    'destination_road_id': row.get('destination_road_id'),
                }
            ))
        
        logger.info(f"✅ Loaded {len(trajectories)} trajectories")
        return trajectories
    
    def load_real_trajectories(self, csv_path: Path, od_type: str, sample_size: int = 100) -> List[Trajectory]:
        """Load real trajectories from train/test CSV (sample for memory efficiency)"""
        if sample_size is None:
            logger.info(f"📂 Loading ALL real trajectories from {csv_path.name}")
            df = pl.read_csv(csv_path)
        else:
            logger.info(f"📂 Loading real trajectories from {csv_path.name} (sampling {sample_size})")
            # Read with sampling
            df = pl.read_csv(csv_path, n_rows=sample_size * 10)  # Read more to account for failures
        
        trajectories = []
        for row in df.iter_rows(named=True):
            # Parse road_id sequence (column name is 'rid_list' in real data)
            road_id_str = row.get('rid_list') or row.get('trace_road_id', '')
            try:
                road_ids = ast.literal_eval(road_id_str)
            except (ValueError, SyntaxError):
                continue
            
            # Convert to GPS coordinates
            coords = self._road_ids_to_coords(road_ids)
            if not coords:
                continue
            
            # Calculate length
            length = self._calculate_length(coords)
            
            trajectories.append(Trajectory(
                road_ids=road_ids,
                coords=coords,
                source="real",
                model="real",
                od_type=od_type,
                length=length,
                metadata={}
            ))
            
            if sample_size is not None and len(trajectories) >= sample_size:
                break
        
        logger.info(f"✅ Loaded {len(trajectories)} real trajectories")
        return trajectories
    
    def _road_ids_to_coords(self, road_ids: List[int]) -> List[Tuple[float, float]]:
        """Convert sequence of road IDs to GPS coordinates"""
        coords = []
        for road_id in road_ids:
            if road_id in self.road_network:
                road_coords = self.road_network[road_id]
                # Add first point of each road (avoids duplication at junctions)
                if road_coords:
                    coords.append(road_coords[0])
        
        # Add last point of last road for completion
        if road_ids and road_ids[-1] in self.road_network:
            last_road = self.road_network[road_ids[-1]]
            if last_road and len(last_road) > 1:
                coords.append(last_road[-1])
        
        return coords
    
    def _calculate_length(self, coords: List[Tuple[float, float]]) -> float:
        """Calculate trajectory length (simple Euclidean distance in degrees)"""
        if len(coords) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(coords) - 1):
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[i + 1]
            total_length += np.sqrt((lon2 - lon1)**2 + (lat2 - lat1)**2)
        
        return total_length


# =============================================================================
# Trajectory Sampler
# =============================================================================

class TrajectorySampler:
    """Sample trajectories using different strategies"""
    
    def __init__(self, config: VisualizationConfig, scenario_results: dict = None):
        self.config = config
        self.scenario_results = scenario_results
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    def sample(self, trajectories: List[Trajectory], strategy: str = None) -> Dict[str, Trajectory]:
        """Sample trajectories using specified strategy"""
        if not trajectories:
            return {}
        
        strategy = strategy or self.config.sample_strategy
        
        if strategy == "random":
            return self.sample_random(trajectories)
        elif strategy == "length_based":
            return self.sample_by_length(trajectories)
        elif strategy == "representative":
            return self.sample_representative(trajectories)
        elif strategy == "scenario":
            return self.sample_by_scenario(trajectories)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
    
    def sample_random(self, trajectories: List[Trajectory]) -> Dict[str, Trajectory]:
        """Randomly sample trajectories"""
        sampled = random.sample(trajectories, min(self.config.samples_per_type, len(trajectories)))
        return {f"sample_{i}": traj for i, traj in enumerate(sampled)}
    
    def sample_by_length(self, trajectories: List[Trajectory]) -> Dict[str, Trajectory]:
        """Sample short (25th), medium (50th), and long (75th) percentile trajectories"""
        if not trajectories:
            return {}
        
        # Calculate length percentiles
        lengths = [traj.length for traj in trajectories]
        p25 = np.percentile(lengths, 25)
        p50 = np.percentile(lengths, 50)
        p75 = np.percentile(lengths, 75)
        
        # Find trajectories closest to each percentile
        result = {}
        
        # Short
        short_idx = min(range(len(lengths)), key=lambda i: abs(lengths[i] - p25))
        result["short"] = trajectories[short_idx]
        
        # Medium
        medium_idx = min(range(len(lengths)), key=lambda i: abs(lengths[i] - p50))
        result["medium"] = trajectories[medium_idx]
        
        # Long
        long_idx = min(range(len(lengths)), key=lambda i: abs(lengths[i] - p75))
        result["long"] = trajectories[long_idx]
        
        logger.info(f"📊 Sampled trajectories: short={p25:.4f}, medium={p50:.4f}, long={p75:.4f}")
        return result
    
    def sample_representative(self, trajectories: List[Trajectory]) -> Dict[str, Trajectory]:
        """Sample median-length trajectories (most representative)"""
        if not trajectories:
            return {}
        
        lengths = [traj.length for traj in trajectories]
        median_length = np.median(lengths)
        
        # Find closest to median
        median_idx = min(range(len(lengths)), key=lambda i: abs(lengths[i] - median_length))
        
        return {"representative": trajectories[median_idx]}
    
    def sample_by_scenario(self, trajectories: List[Trajectory]) -> Dict[str, Trajectory]:
        """Sample representative trajectories from each scenario"""
        if not self.scenario_results or not trajectories:
            logger.warning("No scenario results available, falling back to length-based sampling")
            return self.sample_by_length(trajectories)
        
        # Check if we have trajectory mapping
        trajectory_mapping = self.scenario_results.get('trajectory_mapping', {})
        if not trajectory_mapping or 'trajectories' not in trajectory_mapping:
            logger.warning("No trajectory mapping found, falling back to length-based sampling")
            return self.sample_by_length(trajectories)
        
        # Get scenario distribution from results
        overview = self.scenario_results.get('overview', {})
        scenario_dist = overview.get('scenario_distribution', {})
        
        if not scenario_dist:
            logger.warning("No scenario distribution found, falling back to length-based sampling")
            return self.sample_by_length(trajectories)
        
        # Sort scenarios by count (most common first)
        sorted_scenarios = sorted(scenario_dist.items(), key=lambda x: x[1], reverse=True)
        
        # Take top N scenarios
        top_scenarios = [s[0] for s in sorted_scenarios[:self.config.max_scenarios_to_plot]]
        
        logger.info(f"Sampling from top {len(top_scenarios)} scenarios: {', '.join(top_scenarios)}")
        
        # Group trajectories by scenario
        scenario_groups = {scenario: [] for scenario in top_scenarios}
        
        for traj_info in trajectory_mapping['trajectories']:
            idx = traj_info['index']
            if idx < len(trajectories):
                traj = trajectories[idx]
                # Add trajectory to each scenario it belongs to
                for scenario in traj_info['scenarios']:
                    if scenario in scenario_groups:
                        scenario_groups[scenario].append(traj)
        
        # Sample one representative trajectory from each scenario (median length)
        sampled = {}
        for scenario in top_scenarios:
            trajs = scenario_groups[scenario]
            if trajs:
                # Get median-length trajectory from this scenario
                lengths = [t.length for t in trajs]
                median_length = np.median(lengths)
                median_idx = min(range(len(lengths)), key=lambda i: abs(lengths[i] - median_length))
                sampled[scenario] = trajs[median_idx]
                logger.info(f"  📍 {scenario}: sampled trajectory with length {trajs[median_idx].length:.4f}")
        
        if not sampled:
            logger.warning("No trajectories found for top scenarios, falling back to length-based sampling")
            return self.sample_by_length(trajectories)
        
        return sampled
    


# =============================================================================
# Basemap Manager
# =============================================================================

class BasemapManager:
    """Smart basemap handling with China-friendly providers and fast failure"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.basemap_available = None  # None = untested, True = works, False = failed
        self.provider = self._get_provider()
        
    def _get_provider(self):
        """Get appropriate tile provider based on style"""
        if self.config.basemap_style == "cartodb":
            # CartoDB uses WGS84/EPSG:4326 (matches your data)
            return cx.providers.CartoDB.Positron
        elif self.config.basemap_style == "osm":
            # Standard OSM uses WGS84/EPSG:4326 (matches your data)
            return cx.providers.OpenStreetMap.Mapnik
        elif self.config.basemap_style == "gaode":
            # Gaode uses GCJ-02 coordinates (causes ~50-500m offset from WGS84)
            logger.warning("⚠️  Gaode uses GCJ-02 coordinates, expect ~50-500m offset from WGS84 data")
            try:
                return cx.providers.Gaode.Normal
            except AttributeError:
                logger.warning("⚠️  Gaode provider not available, falling back to CartoDB")
                return cx.providers.CartoDB.Positron
        else:
            return None
    
    def test_connectivity(self) -> bool:
        """Quick connectivity test with a single tile request"""
        if not self.config.basemap_test_first or self.provider is None:
            return False
            
        try:
            import urllib.request
            import socket
            
            # Quick test: try to connect to tile server
            socket.setdefaulttimeout(self.config.basemap_timeout)
            
            # Extract base URL from provider
            provider_url = self.provider.get("url", "")
            if "{s}" in provider_url:
                provider_url = provider_url.replace("{s}", "a")
            
            # Try to fetch a test tile (zoom 0, x=0, y=0)
            test_url = provider_url.replace("{z}", "0").replace("{x}", "0").replace("{y}", "0")
            
            req = urllib.request.Request(test_url, headers={'User-Agent': 'Mozilla/5.0'})
            urllib.request.urlopen(req, timeout=self.config.basemap_timeout)
            
            return True
        except Exception as e:
            logger.info(f"🌐 Basemap connectivity test failed: {type(e).__name__}")
            return False
    
    def add_basemap_safe(self, ax, **kwargs) -> bool:
        """Add basemap with error handling and fast timeout"""
        if self.provider is None or self.config.basemap_style == "none":
            return False
        
        # If already tested and failed, skip
        if self.basemap_available is False:
            return False
        
        # Test connectivity on first attempt
        if self.basemap_available is None:
            logger.info(f"🌐 Testing {self.config.basemap_style} basemap connectivity...")
            self.basemap_available = self.test_connectivity()
            
            if not self.basemap_available:
                logger.info("🗺️  Basemap unavailable, continuing without basemap")
                return False
        
        # Try to add basemap with timeout
        try:
            # Contextily doesn't directly support timeout, but we can set socket timeout
            import socket
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(self.config.basemap_timeout)
            
            try:
                cx.add_basemap(ax, source=self.provider, **kwargs)
                return True
            finally:
                socket.setdefaulttimeout(old_timeout)
                
        except Exception as e:
            # On any error, disable basemap for remaining plots
            if self.basemap_available is not False:
                logger.info(f"⚠️  Basemap failed ({type(e).__name__}), disabling for remaining plots")
                self.basemap_available = False
            return False


# =============================================================================
# Trajectory Comparison Plotter (Shared logic for cross-model comparisons)
# =============================================================================

class TrajectoryComparisonPlotter:
    """Shared logic for comparing trajectories across models"""
    
    def __init__(self, config: VisualizationConfig, road_network: Dict):
        self.config = config
        self.road_network = road_network
        
        # Model styling constants
        self.model_colors = {
            'vanilla': '#e74c3c',          # Red
            'distilled': '#3498db',        # Blue
            'distilled_seed44': '#2ecc71', # Green
            'real': '#f39c12',             # Orange/Gold
        }
        
        self.model_linestyles = {
            'vanilla': '-',
            'distilled': '-',
            'distilled_seed44': '-',
            'real': '--',  # Dashed for real
        }
        
        self.model_labels = {
            'vanilla': 'Vanilla',
            'distilled': 'Distilled (seed 42)',
            'distilled_seed44': 'Distilled (seed 44)',
            'real': 'Real Trajectory',
        }
    
    def plot_comparison(self, trajectories: Dict[str, 'Trajectory'], 
                       output_path: Path, title: str, 
                       scenario_labels: Dict[str, List[str]] = None):
        """Core comparison plotting logic with enhanced scenario labels
        
        Args:
            trajectories: Dict mapping model_name -> Trajectory
            output_path: Path for output files (without extension)
            title: Plot title
            scenario_labels: Dict mapping model_name -> list of scenario tags
        """
        if not trajectories:
            logger.warning("No trajectories to plot")
            return
        
        fig, ax = plt.subplots(figsize=self.config.figsize, facecolor='white')
        ax.set_facecolor('white')
        
        # Collect all coordinates for bounds
        all_lons, all_lats = [], []
        
        # Plot each model's trajectory
        for model_name in sorted(trajectories.keys(), key=lambda x: (x != 'real', x)):
            traj = trajectories[model_name]
            if not traj.coords:
                continue
            
            lons = [c[0] for c in traj.coords]
            lats = [c[1] for c in traj.coords]
            
            all_lons.extend(lons)
            all_lats.extend(lats)
            
            # Plot trajectory
            ax.plot(lons, lats,
                   color=self.model_colors.get(model_name, '#95a5a6'),
                   linestyle=self.model_linestyles.get(model_name, '-'),
                   linewidth=2.5 if model_name == 'real' else 2,
                   alpha=0.8,
                   zorder=10 if model_name == 'real' else 5)
            
            # Mark start and end
            ax.scatter(lons[0], lats[0], c=self.model_colors.get(model_name, '#95a5a6'),
                      marker='o', s=100, zorder=15, edgecolors='white', linewidths=1.5)
            ax.scatter(lons[-1], lats[-1], c=self.model_colors.get(model_name, '#95a5a6'),
                      marker='s', s=100, zorder=15, edgecolors='white', linewidths=1.5)
        
        # Set bounds
        if all_lons and all_lats:
            margin = self.config.margin
            ax.set_xlim(min(all_lons) - margin, max(all_lons) + margin)
            ax.set_ylim(min(all_lats) - margin, max(all_lats) + margin)
        
        # Build legend with scenario labels
        legend_elements = []
        for model_name in sorted(trajectories.keys(), key=lambda x: (x != 'real', x)):
            label = self._build_legend_label(model_name, 
                                             scenario_labels.get(model_name) if scenario_labels else None)
            legend_elements.append(
                plt.Line2D([0], [0], 
                          color=self.model_colors.get(model_name, '#95a5a6'),
                          linestyle=self.model_linestyles.get(model_name, '-'),
                          linewidth=2.5 if model_name == 'real' else 2,
                          label=label)
            )
        
        # Position legend outside plot area
        ax.legend(handles=legend_elements, 
                 loc='center left',
                 bbox_to_anchor=(1.02, 0.5),
                 frameon=True,
                 fontsize=9,
                 title='Models' if not scenario_labels else 'Models (Scenarios)')
        
        ax.set_title(title, fontsize=14, pad=15)
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
        
        # Save
        plt.savefig(f"{output_path}.pdf", dpi=self.config.dpi, bbox_inches='tight')
        plt.savefig(f"{output_path}.png", dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
    
    def _build_legend_label(self, model_name: str, scenarios: List[str] = None) -> str:
        """Build legend label with scenario tags if provided"""
        base_label = self.model_labels.get(model_name, model_name)
        if scenarios:
            # Format scenarios nicely
            scenario_display = [s.replace('_', ' ').title() for s in scenarios[:3]]
            scenario_str = ", ".join(scenario_display)
            if len(scenarios) > 3:
                scenario_str += f" +{len(scenarios)-3}"
            return f"{base_label}\n({scenario_str})"
        return base_label


# =============================================================================
# Trajectory Plotter
# =============================================================================

class TrajectoryPlotter:
    """Plot trajectories on maps with various styles"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.basemap_manager = BasemapManager(config)
        self.road_network = None
        
        # Load road network if needed
        if config.show_road_network:
            self.road_network = self._load_road_network()
    
    def _load_road_network(self) -> Optional[List[List[Tuple[float, float]]]]:
        """Load road network geometry from roadmap.geo"""
        try:
            logger.info(f"📍 Loading road network from {self.config.roadmap_path}")
            df = pl.read_csv(
                self.config.roadmap_path,
                schema_overrides={"lanes": pl.Utf8, "oneway": pl.Utf8}
            )
            
            road_segments = []
            for coords_str in df['coordinates']:
                try:
                    # Parse coordinate string: "[[lon1, lat1], [lon2, lat2], ...]"
                    coords = ast.literal_eval(coords_str)
                    road_segments.append([(lon, lat) for lon, lat in coords])
                except (ValueError, SyntaxError):
                    continue
            
            logger.info(f"✅ Loaded {len(road_segments)} road segments")
            return road_segments
            
        except Exception as e:
            logger.warning(f"⚠️  Could not load road network: {e}")
            return None
    
    def _plot_road_network(self, ax, bounds: Tuple[float, float, float, float] = None, convert_coords: bool = False):
        """Plot road network as light gray lines in background
        
        Args:
            ax: Matplotlib axis
            bounds: (min_lon, max_lon, min_lat, max_lat) to filter roads
            convert_coords: Apply GCJ-02 conversion
        
        Returns:
            True if roads were plotted, False otherwise
        """
        if self.road_network is None:
            return False
        
        # If no bounds provided, plot all roads (slow!)
        if bounds is None:
            segments_to_plot = self.road_network
        else:
            # Filter roads within bounds for faster plotting
            min_lon, max_lon, min_lat, max_lat = bounds
            margin = self.config.margin * 2  # Extra margin for roads
            segments_to_plot = []
            
            for segment in self.road_network:
                if len(segment) < 2:
                    continue
                
                # Check if segment intersects with bounds
                seg_lons = [lon for lon, lat in segment]
                seg_lats = [lat for lon, lat in segment]
                
                if (min(seg_lons) <= max_lon + margin and max(seg_lons) >= min_lon - margin and
                    min(seg_lats) <= max_lat + margin and max(seg_lats) >= min_lat - margin):
                    segments_to_plot.append(segment)
        
        # Plot filtered segments with better visibility
        for i, segment in enumerate(segments_to_plot):
            # Convert coordinates if using Gaode basemap
            if convert_coords:
                segment = self._convert_coords_for_basemap(segment)
            
            lons, lats = zip(*segment)
            # Add label only to first segment for legend
            label = 'Road Network' if i == 0 else None
            ax.plot(lons, lats, color='#CCCCCC', linewidth=0.5, alpha=0.6, zorder=1, label=label)
        
        return True
    
    def _wgs84_to_gcj02(self, lon: float, lat: float) -> tuple[float, float]:
        """Convert WGS84 coordinates to GCJ-02 (for Chinese map providers like Gaode)
        
        Algorithm based on: https://github.com/wandergis/coordtransform
        """
        import math
        
        # Constants
        a = 6378245.0  # Semi-major axis
        ee = 0.00669342162296594323  # Eccentricity squared
        
        def _transform_lat(x, y):
            ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
            ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
            ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
            ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
            return ret
        
        def _transform_lon(x, y):
            ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
            ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
            ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
            ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
            return ret
        
        # Check if coordinates are in China
        if lon < 72.004 or lon > 137.8347 or lat < 0.8293 or lat > 55.8271:
            return lon, lat  # Outside China, no conversion needed
        
        dlat = _transform_lat(lon - 105.0, lat - 35.0)
        dlon = _transform_lon(lon - 105.0, lat - 35.0)
        radlat = lat / 180.0 * math.pi
        magic = math.sin(radlat)
        magic = 1 - ee * magic * magic
        sqrtmagic = math.sqrt(magic)
        dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
        dlon = (dlon * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)
        mglat = lat + dlat
        mglon = lon + dlon
        return mglon, mglat
    
    def _convert_coords_for_basemap(self, coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Convert coordinates if using Gaode basemap (WGS84 -> GCJ-02)"""
        if self.config.basemap_style == "gaode":
            return [self._wgs84_to_gcj02(lon, lat) for lon, lat in coords]
        return coords
        
    def plot_single(self, trajectory: Trajectory, output_path: Path, title: str = None):
        """Plot a single trajectory"""
        if not trajectory.coords:
            logger.warning("Empty trajectory, skipping plot")
            return
        
        fig, ax = plt.subplots(figsize=self.config.figsize, facecolor='white')
        ax.set_facecolor('white')
        
        # Convert coordinates for basemap if needed (WGS84 -> GCJ-02 for Gaode)
        plot_coords = self._convert_coords_for_basemap(trajectory.coords)
        lons, lats = zip(*plot_coords)
        
        # Calculate bounds for road network filtering
        margin = self.config.margin
        bounds = (min(lons), max(lons), min(lats), max(lats))
        
        # Plot road network in background if enabled (with bounds for speed)
        convert_coords = self.config.basemap_style == "gaode"
        self._plot_road_network(ax, bounds=bounds, convert_coords=convert_coords)
        
        # Plot trajectory line
        ax.plot(lons, lats, 'b-', linewidth=2.5, label='Trajectory', zorder=3, alpha=0.8)
        
        # Start marker (green circle)
        ax.scatter(lons[0], lats[0], c='green', s=150, marker='o', 
                   label='Start', zorder=4, edgecolors='black', linewidths=2)
        
        # End marker (red square)
        ax.scatter(lons[-1], lats[-1], c='red', s=150, marker='s', 
                   label='End', zorder=4, edgecolors='black', linewidths=2)
        
        # Auto-zoom with padding
        ax.set_xlim(min(lons) - margin, max(lons) + margin)
        ax.set_ylim(min(lats) - margin, max(lats) + margin)
        
        # Styling
        if title is None:
            title = f"{trajectory.model} - {trajectory.od_type} OD - {trajectory.source}"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # Save both formats
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{output_path}.pdf", dpi=self.config.dpi, bbox_inches='tight')
        plt.savefig(f"{output_path}.png", dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Saved: {output_path}.{{pdf,png}}")
    
    def plot_overlaid(self, trajectories: Dict[str, Trajectory], output_path: Path, title: str = None):
        """Plot multiple trajectories overlaid on same map with color coding"""
        if not trajectories:
            logger.warning("No trajectories to plot")
            return
        
        # Color scheme for different trajectory types
        colors = {
            'short': '#1f77b4',    # Blue
            'medium': '#ff7f0e',   # Orange
            'long': '#2ca02c',     # Green
            'train': '#1f77b4',
            'test': '#ff7f0e',
            'generated': '#2ca02c',
        }
        
        fig, ax = plt.subplots(figsize=self.config.figsize, facecolor='white')
        ax.set_facecolor('white')
        
        # Collect all coordinates for bounds calculation
        all_lons, all_lats = [], []
        
        # Plot each trajectory
        for label, traj in trajectories.items():
            if not traj.coords:
                continue
            
            # Convert coordinates for basemap if needed (WGS84 -> GCJ-02 for Gaode)
            plot_coords = self._convert_coords_for_basemap(traj.coords)
            lons, lats = zip(*plot_coords)
            all_lons.extend(lons)
            all_lats.extend(lats)
            
            color = colors.get(label, '#333333')
            
            # Plot line
            ax.plot(lons, lats, '-', linewidth=2.5, label=f'{label.capitalize()}', 
                    color=color, zorder=3, alpha=0.7)
            
            # Start marker
            ax.scatter(lons[0], lats[0], c=color, s=100, marker='o', 
                       zorder=4, edgecolors='black', linewidths=1.5, alpha=0.9)
            
            # End marker
            ax.scatter(lons[-1], lats[-1], c=color, s=100, marker='s', 
                       zorder=4, edgecolors='black', linewidths=1.5, alpha=0.9)
        
        # Auto-zoom with padding and plot road network
        if all_lons and all_lats:
            margin = self.config.margin
            bounds = (min(all_lons), max(all_lons), min(all_lats), max(all_lats))
            
            # Plot road network in background if enabled
            convert_coords = self.config.basemap_style == "gaode"
            self._plot_road_network(ax, bounds=bounds, convert_coords=convert_coords)
            
            ax.set_xlim(min(all_lons) - margin, max(all_lons) + margin)
            ax.set_ylim(min(all_lats) - margin, max(all_lats) + margin)
        
        # Styling
        if title is None:
            first_traj = next(iter(trajectories.values()))
            title = f"{first_traj.model} - {first_traj.od_type} OD - All Lengths"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # Save both formats
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{output_path}.pdf", dpi=self.config.dpi, bbox_inches='tight')
        plt.savefig(f"{output_path}.png", dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Saved: {output_path}.{{pdf,png}}")
    
    def _calculate_trajectory_overlap(self, trajectories: Dict[str, Trajectory]) -> Dict[str, float]:
        """Calculate overlap percentage for each trajectory with others"""
        overlaps = {}
        
        for model_name, traj in trajectories.items():
            if not traj.road_ids:
                overlaps[model_name] = 0.0
                continue
            
            # Get road IDs for this trajectory
            traj_roads = set(traj.road_ids)
            
            # Get all road IDs from other trajectories
            other_roads = set()
            for other_model, other_traj in trajectories.items():
                if other_model != model_name and other_traj.road_ids:
                    other_roads.update(other_traj.road_ids)
            
            # Calculate overlap percentage
            if traj_roads and other_roads:
                overlap_count = len(traj_roads.intersection(other_roads))
                overlap_pct = (overlap_count / len(traj_roads)) * 100
                overlaps[model_name] = overlap_pct
            else:
                overlaps[model_name] = 0.0
        
        return overlaps
    
    def plot_cross_model_comparison(self, trajectories: Dict[str, Trajectory], output_path: Path, 
                                   title: str = None, missing_models: List[str] = None):
        """Plot trajectories from different models for the same OD pair"""
        if not trajectories:
            logger.warning("No trajectories to plot")
            return
        
        if missing_models is None:
            missing_models = []
        
        # Calculate overlap percentages
        overlaps = self._calculate_trajectory_overlap(trajectories)
        
        # Color scheme for different models
        model_colors = {
            'vanilla': '#e74c3c',          # Red
            'distilled': '#3498db',        # Blue
            'distilled_seed44': '#2ecc71', # Green
            'real': '#f39c12',             # Orange/Gold
        }
        
        # Line styles
        model_linestyles = {
            'vanilla': '-',
            'distilled': '-',
            'distilled_seed44': '-',
            'real': '--',  # Dashed for real
        }
        
        # Display names
        model_labels = {
            'vanilla': 'Vanilla',
            'distilled': 'Distilled (seed 42)',
            'distilled_seed44': 'Distilled (seed 44)',
            'real': 'Real Trajectory',
        }
        
        fig, ax = plt.subplots(figsize=self.config.figsize, facecolor='white')
        ax.set_facecolor('white')
        
        # Collect all coordinates for bounds calculation
        all_lons, all_lats = [], []
        
        # Plot each model's trajectory
        for model_name in sorted(trajectories.keys(), key=lambda x: (x != 'real', x)):
            traj = trajectories[model_name]
            if not traj.coords:
                continue
            
            # Convert coordinates for basemap if needed (WGS84 -> GCJ-02 for Gaode)
            plot_coords = self._convert_coords_for_basemap(traj.coords)
            lons, lats = zip(*plot_coords)
            all_lons.extend(lons)
            all_lats.extend(lats)
            
            color = model_colors.get(model_name, '#333333')
            linestyle = model_linestyles.get(model_name, '-')
            label = model_labels.get(model_name, model_name)
            
            # Different line width for real trajectory
            linewidth = 3.5 if model_name == 'real' else 2.5
            
            # Plot line with reduced opacity for visibility when overlapping
            ax.plot(lons, lats, linestyle=linestyle, linewidth=linewidth, 
                   label=label, color=color, zorder=4 if model_name == 'real' else 3, alpha=0.6)
            
            # Start marker
            ax.scatter(lons[0], lats[0], c=color, s=120, marker='o', 
                      zorder=5, edgecolors='black', linewidths=1.5, alpha=0.9)
            
            # End marker
            ax.scatter(lons[-1], lats[-1], c=color, s=120, marker='s', 
                      zorder=5, edgecolors='black', linewidths=1.5, alpha=0.9)
        
        # Auto-zoom with padding and plot road network
        if all_lons and all_lats:
            margin = self.config.margin
            bounds = (min(all_lons), max(all_lons), min(all_lats), max(all_lats))
            
            # Plot road network in background if enabled
            convert_coords = self.config.basemap_style == "gaode"
            self._plot_road_network(ax, bounds=bounds, convert_coords=convert_coords)
            
            ax.set_xlim(min(all_lons) - margin, max(all_lons) + margin)
            ax.set_ylim(min(all_lats) - margin, max(all_lats) + margin)
        
        # Styling
        if title is None:
            title = "Cross-Model Comparison - Same OD Pair"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Create custom legend with start/end markers
        from matplotlib.lines import Line2D
        
        legend_elements = []
        
        # Add model trajectories to legend with overlap percentages (except real)
        for model_name in sorted(trajectories.keys(), key=lambda x: (x != 'real', x)):
            color = model_colors.get(model_name, '#333333')
            linestyle = model_linestyles.get(model_name, '-')
            base_label = model_labels.get(model_name, model_name)
            linewidth = 3.5 if model_name == 'real' else 2.5
            
            # Add overlap percentage to label (skip for real trajectory)
            if model_name == 'real':
                label = base_label
            else:
                overlap_pct = overlaps.get(model_name, 0.0)
                label = f"{base_label} ({overlap_pct:.1f}% overlap)"
            
            legend_elements.append(Line2D([0], [0], color=color, linewidth=linewidth, 
                                         linestyle=linestyle, label=label))
        
        # Add separator
        legend_elements.append(Line2D([0], [0], color='none', label=''))
        
        # Add road network
        if self.road_network:
            legend_elements.append(Line2D([0], [0], color='#CCCCCC', linewidth=2, 
                                         alpha=0.6, label='Road Network'))
        
        # Add start/end markers
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor='gray', markersize=10, 
                                     markeredgecolor='black', markeredgewidth=1.5,
                                     label='Start Point', linestyle=''))
        legend_elements.append(Line2D([0], [0], marker='s', color='w', 
                                     markerfacecolor='gray', markersize=10,
                                     markeredgecolor='black', markeredgewidth=1.5,
                                     label='End Point', linestyle=''))
        
        # Place legend outside plot area to avoid obstructing trajectories
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                 fontsize=11, framealpha=0.95, title='Legend', title_fontsize=12)
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        
        # Add info box with missing models warning if applicable
        info_lines = [f"Trajectories: {len(trajectories)}/{len(model_labels)}"]
        if missing_models:
            missing_names = [model_labels.get(m, m) for m in missing_models]
            info_lines.append(f"⚠️  Missing: {', '.join(missing_names)}")
        
        info_text = '\n'.join(info_lines)
        bbox_props = dict(boxstyle='round', facecolor='lightyellow' if missing_models else 'white', 
                         alpha=0.9, edgecolor='orange' if missing_models else 'gray')
        ax.text(0.02, 0.02, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='bottom', bbox=bbox_props)
        
        # Save both formats
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{output_path}.pdf", dpi=self.config.dpi, bbox_inches='tight')
        plt.savefig(f"{output_path}.png", dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Saved cross-model comparison: {output_path}.{{pdf,png}}")


# =============================================================================
# Main Orchestrator
# =============================================================================

class TrajectoryVisualizer:
    """Main orchestrator for trajectory visualization"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        
        # Load road network
        road_loader = RoadNetworkLoader(config.roadmap_path)
        self.road_network = road_loader.load()
        
        # Load scenario results if available
        self.scenario_results = self._load_scenario_results()
        if self.scenario_results:
            logger.info("✅ Loaded scenario analysis results")
        
        # Initialize components
        self.loader = TrajectoryLoader(config, self.road_network)
        self.sampler = TrajectorySampler(config, self.scenario_results)
        self.plotter = TrajectoryPlotter(config)
        self.comparison_plotter = TrajectoryComparisonPlotter(config, self.road_network)
    
    def _load_scenario_results(self) -> Optional[dict]:
        """Load scenario analysis results if available"""
        scenarios_dir = self.config.eval_dir / "scenarios"
        if not scenarios_dir.exists():
            return None
        
        # Find scenario analysis JSON files - look for any model/od_source
        for json_file in scenarios_dir.rglob("scenario_analysis.json"):
            try:
                with open(json_file) as f:
                    results = json.load(f)
                
                # Also load trajectory mapping if available
                mapping_file = json_file.parent / "trajectory_scenarios.json"
                if mapping_file.exists():
                    with open(mapping_file) as f:
                        results['trajectory_mapping'] = json.load(f)
                
                return results
            except Exception as e:
                logger.warning(f"Failed to load scenario results from {json_file}: {e}")
        
        return None
        
    def run(self):
        """Run the complete visualization pipeline"""
        logger.info("🚀 Starting trajectory visualization pipeline")
        
        # Detect generated trajectory files
        gene_files = self._detect_gene_files()
        
        if not gene_files:
            logger.error("❌ No generated trajectory files found")
            return
        
        # Process each model individually
        if self.config.generate_separate or self.config.generate_overlaid:
            for model_info in gene_files:
                self._process_model(model_info)
        
        # Cross-model comparison for same OD pairs
        if self.config.generate_cross_model:
            self._generate_cross_model_comparisons(gene_files)
        
        # Scenario-based cross-model comparison
        if self.config.generate_scenario_cross_model:
            if self.scenario_results:
                self._generate_scenario_cross_model_comparisons(gene_files)
            else:
                logger.warning("⚠️  Scenario cross-model mode requires scenario analysis results")
        
        logger.info("✅ Visualization pipeline completed!")
    
    def _detect_gene_files(self) -> List[Dict]:
        """Detect generated trajectory CSV files"""
        logger.info(f"🔍 Detecting generated trajectory files in {self.config.gene_dir}")
        
        gene_files = []
        
        # Search recursively for CSV files (handles seed subdirectories)
        for csv_file in self.config.gene_dir.rglob("*.csv"):
            # Parse filename to extract model and OD type
            filename = csv_file.name
            
            # Skip old unnamed files
            if not any(m in filename for m in ['distilled', 'vanilla']):
                continue
            
            model = None
            od_type = None
            
            if 'distilled_seed44' in filename:
                model = 'distilled_seed44'
            elif 'distilled' in filename:
                model = 'distilled'
            elif 'vanilla' in filename:
                model = 'vanilla'
            
            if 'train' in filename:
                od_type = 'train'
            elif 'test' in filename:
                od_type = 'test'
            
            if model and od_type:
                gene_files.append({
                    'path': csv_file,
                    'model': model,
                    'od_type': od_type
                })
                logger.info(f"  Found: {model} - {od_type} OD")
        
        return gene_files
    
    def _process_model(self, model_info: Dict):
        """Process trajectories for a single model"""
        model = model_info['model']
        od_type = model_info['od_type']
        csv_path = model_info['path']
        
        logger.info(f"\n📊 Processing: {model} - {od_type} OD")
        
        # Load trajectories
        trajectories = self.loader.load_generated_trajectories(csv_path, model, od_type)
        
        if not trajectories:
            logger.warning(f"⚠️  No valid trajectories found for {model} - {od_type}")
            return
        
        # Sample trajectories
        sampled = self.sampler.sample(trajectories)
        
        if not sampled:
            logger.warning(f"⚠️  No trajectories sampled for {model} - {od_type}")
            return
        
        # Generate separate plots
        if self.config.generate_separate:
            self._generate_separate_plots(sampled, model, od_type)
        
        # Generate overlaid plot
        if self.config.generate_overlaid:
            self._generate_overlaid_plot(sampled, model, od_type)
    
    def _generate_separate_plots(self, sampled: Dict[str, Trajectory], model: str, od_type: str):
        """Generate separate plots for each trajectory"""
        logger.info("📊 Generating separate plots...")
        
        output_dir = self.config.output_dir / "separate"
        
        for label, traj in sampled.items():
            output_path = output_dir / f"{model}_{od_type}_{label}"
            title = f"{model.replace('_', ' ').title()} - {od_type.upper()} OD - {label.capitalize()}"
            self.plotter.plot_single(traj, output_path, title=title)
    
    def _generate_overlaid_plot(self, sampled: Dict[str, Trajectory], model: str, od_type: str):
        """Generate overlaid plot with all sampled trajectories"""
        logger.info("📊 Generating overlaid plot...")
        
        output_dir = self.config.output_dir / "overlaid"
        output_path = output_dir / f"{model}_{od_type}_all"
        title = f"{model.replace('_', ' ').title()} - {od_type.upper()} OD - All Sampled Trajectories"
        
        self.plotter.plot_overlaid(sampled, output_path, title=title)
    
    def _generate_cross_model_comparisons(self, gene_files: List[Dict]):
        """Generate cross-model comparisons for same OD pairs"""
        logger.info("\n🔄 Generating cross-model comparisons for same OD pairs...")
        
        # Group by OD type
        for od_type in ['train', 'test']:
            logger.info(f"\n📊 Processing {od_type.upper()} OD comparisons...")
            
            # Load all models for this OD type
            models_data = {}
            for model_info in gene_files:
                if model_info['od_type'] == od_type:
                    model = model_info['model']
                    csv_path = model_info['path']
                    
                    logger.info(f"  Loading {model}...")
                    trajectories = self.loader.load_generated_trajectories(csv_path, model, od_type)
                    
                    if trajectories:
                        models_data[model] = trajectories
            
            # Optionally load real trajectories for comparison
            if self.config.include_real_in_cross_model:
                logger.info(f"  Loading real {od_type} trajectories...")
                real_csv = self.config.train_csv_path if od_type == 'train' else self.config.test_csv_path
                # Load much larger sample to capture all generated OD pairs
                # Generated trajectories use OD pairs from real data, so they should all exist
                real_trajectories = self.loader.load_real_trajectories(real_csv, od_type, sample_size=50000)
                
                if real_trajectories:
                    models_data['real'] = real_trajectories
                    logger.info(f"  ✅ Loaded {len(real_trajectories)} real trajectories")
            
            if len(models_data) < 2:
                logger.warning(f"⚠️  Not enough models loaded for {od_type} OD comparison")
                continue
            
            # Find common OD pairs
            self._plot_matching_od_pairs(models_data, od_type)
    
    def _load_all_models_for_od(self, gene_files: List[Dict], od_type: str) -> Dict[str, List['Trajectory']]:
        """Load all models and optionally real trajectories for a given OD type"""
        models_data = {}
        
        # Load generated models
        for model_info in gene_files:
            if model_info['od_type'] == od_type:
                model = model_info['model']
                csv_path = model_info['path']
                
                logger.info(f"  Loading {model}...")
                trajectories = self.loader.load_generated_trajectories(csv_path, model, od_type)
                
                if trajectories:
                    models_data[model] = trajectories
        
        # Optionally load real trajectories
        if self.config.include_real_in_cross_model:
            logger.info(f"  Loading real {od_type} trajectories...")
            real_csv = self.config.train_csv_path if od_type == 'train' else self.config.test_csv_path
            # Load all data for better OD pair coverage and route variation detection
            real_trajectories = self.loader.load_real_trajectories(real_csv, od_type, sample_size=None)
            
            if real_trajectories:
                models_data['real'] = real_trajectories
                logger.info(f"  ✅ Loaded {len(real_trajectories)} real trajectories")
        
        return models_data
    
    def _plot_matching_od_pairs(self, models_data: Dict[str, List[Trajectory]], od_type: str, 
                                scenario: str = None, max_plots: int = None):
        """Find and plot trajectories with matching OD pairs across models
        
        Args:
            models_data: Dict mapping model_name -> list of trajectories
            od_type: 'train' or 'test'
            scenario: Optional scenario name for scenario-based comparisons
            max_plots: Optional maximum number of plots to generate (default: 10 for scenarios, no limit otherwise)
        """
        
        # Adjust output path and title based on scenario
        if scenario:
            output_dir = self.config.output_dir / "scenario_cross_model" / od_type / scenario
            title_prefix = f"{scenario.replace('_', ' ').title()} Scenario - "
            if max_plots is None:
                max_plots = 10  # Default limit for scenario-based comparisons
        else:
            output_dir = self.config.output_dir / "cross_model" / od_type
            title_prefix = ""
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build OD pair index for each model
        od_indices = {}
        for model_name, trajectories in models_data.items():
            od_indices[model_name] = {}
            for traj in trajectories:
                if traj.road_ids and len(traj.road_ids) >= 2:
                    origin = traj.road_ids[0]
                    destination = traj.road_ids[-1]
                    od_pair = (origin, destination)
                    
                    if od_pair not in od_indices[model_name]:
                        od_indices[model_name][od_pair] = []
                    od_indices[model_name][od_pair].append(traj)
        
        # Find OD pairs that appear in ALL models (including real)
        all_od_pairs = set(od_indices[list(models_data.keys())[0]].keys())
        for model_name in list(models_data.keys())[1:]:
            all_od_pairs &= set(od_indices[model_name].keys())
        
        if not all_od_pairs:
            logger.warning(f"⚠️  No common OD pairs found across all models for {od_type} OD")
            return
        
        logger.info(f"✅ Found {len(all_od_pairs)} common OD pairs across all models")
        
        # Sample representative OD pairs (including edge cases)
        sampled_od_pairs = self._sample_od_pairs_for_comparison(list(all_od_pairs), od_indices, models_data)
        
        # Apply max_plots limit if specified
        if max_plots is not None and len(sampled_od_pairs) > max_plots:
            sampled_od_pairs = sampled_od_pairs[:max_plots]
            logger.info(f"📊 Selected {len(sampled_od_pairs)} OD pairs for visualization (limited to max {max_plots})")
        else:
            logger.info(f"📊 Selected {len(sampled_od_pairs)} OD pairs for visualization (captures edge cases from analysis)")
        
        # Generate comparison plots
        for i, od_pair in enumerate(sampled_od_pairs):
            origin, destination = od_pair
            
            # Collect one trajectory from each model for this OD pair
            comparison_trajs = {}
            missing_models = []
            
            for model_name in sorted(models_data.keys()):
                if od_pair in od_indices[model_name] and od_indices[model_name][od_pair]:
                    trajs = od_indices[model_name][od_pair]
                    # Pick the first (or median-length) trajectory
                    comparison_trajs[model_name] = trajs[0]
                else:
                    missing_models.append(model_name)
            
            # Generate the comparison plot
            output_path = output_dir / f"{od_type}_od_comparison_{i+1}_origin{origin}_dest{destination}"
            title = f"{title_prefix}{od_type.upper()} OD: All Models - Origin {origin} → Destination {destination}"
            
            self.plotter.plot_cross_model_comparison(comparison_trajs, output_path, title=title, 
                                                     missing_models=missing_models)
            logger.info(f"  ✅ Saved comparison {i+1}/{len(sampled_od_pairs)}")
    
    def _sample_od_pairs_for_comparison(self, od_pairs: List[Tuple[int, int]], 
                                       od_indices: Dict, models_data: Dict) -> List[Tuple[int, int]]:
        """Sample representative OD pairs for comparison (short, medium, long)"""
        
        # Calculate average trajectory length for each OD pair
        od_lengths = {}
        for od_pair in od_pairs:
            lengths = []
            for model_name, trajs in od_indices.items():
                if od_pair in trajs and trajs[od_pair]:
                    lengths.append(trajs[od_pair][0].length)
            if lengths:
                od_lengths[od_pair] = sum(lengths) / len(lengths)
        
        if not od_lengths:
            return od_pairs[:3]  # Fallback
        
        # Sort by length
        sorted_od_pairs = sorted(od_lengths.items(), key=lambda x: x[1])
        
        # Sample multiple representative lengths to capture edge cases from analysis
        n = len(sorted_od_pairs)
        if n >= 10:
            # Sample 10+ trajectories across the full range:
            # - Extremes: 0th (shortest), 100th (longest) - edge cases
            # - Key percentiles: 5th, 10th, 25th, 50th, 75th, 90th, 95th
            # - Extra coverage: 33rd, 66th
            percentiles = [0.05, 0.10, 0.25, 0.33, 0.50, 0.66, 0.75, 0.90, 0.95]
            indices = [int(n * p) for p in percentiles]
            # Add absolute extremes
            indices = [0] + indices + [n - 1]
            return [sorted_od_pairs[i][0] for i in indices]
        elif n >= 6:
            # Sample 6 trajectories: very short, short, medium-short, medium, medium-long, long
            percentiles = [0.10, 0.25, 0.40, 0.60, 0.75, 0.90]
            indices = [int(n * p) for p in percentiles]
            return [sorted_od_pairs[i][0] for i in indices]
        elif n >= 5:
            # Sample 5 trajectories: extremes + quartiles
            indices = [0, int(n * 0.25), int(n * 0.50), int(n * 0.75), n - 1]
            return [sorted_od_pairs[i][0] for i in indices]
        elif n >= 3:
            # Fallback to short, medium, long
            indices = [int(n * 0.25), int(n * 0.50), int(n * 0.75)]
            return [sorted_od_pairs[i][0] for i in indices]
        else:
            # Return all available
            return [pair for pair, _ in sorted_od_pairs]
    
    def _track_od_pairs_for_multi_scenario(self, models_data: Dict, scenario: str, 
                                           od_scenario_map: Dict):
        """Track OD pairs and their trajectories across scenarios for multi-scenario plots"""
        
        # Find common OD pairs across all models in this scenario
        od_indices = {}
        for model_name, trajectories in models_data.items():
            od_indices[model_name] = {}
            for traj in trajectories:
                if traj.road_ids and len(traj.road_ids) >= 2:
                    origin = traj.road_ids[0]
                    destination = traj.road_ids[-1]
                    od_pair = (origin, destination)
                    
                    if od_pair not in od_indices[model_name]:
                        od_indices[model_name][od_pair] = []
                    od_indices[model_name][od_pair].append(traj)
        
        # Find OD pairs common to all models
        if not od_indices:
            return
        
        model_names = list(models_data.keys())
        common_od_pairs = set(od_indices[model_names[0]].keys())
        for model_name in model_names[1:]:
            common_od_pairs &= set(od_indices[model_name].keys())
        
        # Store trajectories for each common OD pair in this scenario
        for od_pair in common_od_pairs:
            if od_pair not in od_scenario_map:
                od_scenario_map[od_pair] = {}
            
            # Store one representative trajectory from each model for this OD pair
            od_scenario_map[od_pair][scenario] = {
                model_name: od_indices[model_name][od_pair][0]  # Take first trajectory
                for model_name in model_names
            }
    
    def _has_route_variation(self, scenarios_dict: Dict) -> bool:
        """Check if at least one model takes different routes across scenarios"""
        
        # Get all model names
        first_scenario = list(scenarios_dict.keys())[0]
        model_names = list(scenarios_dict[first_scenario].keys())
        
        # For each model, check if routes differ across scenarios
        for model_name in model_names:
            # Collect all routes (as tuples of road IDs) for this model across scenarios
            routes = []
            for scenario, models_trajs in scenarios_dict.items():
                if model_name in models_trajs:
                    traj = models_trajs[model_name]
                    if traj.road_ids:
                        routes.append(tuple(traj.road_ids))
            
            # If this model has at least 2 different routes, there's variation
            if len(set(routes)) > 1:
                return True
        
        # No model has route variation across scenarios
        return False
    
    def _generate_multi_scenario_comparisons(self, od_scenario_map: Dict, od_type: str):
        """Generate concatenated plots for OD pairs that appear in multiple scenarios"""
        
        # Find OD pairs that appear in 2+ scenarios
        multi_scenario_pairs = {
            od_pair: scenarios 
            for od_pair, scenarios in od_scenario_map.items() 
            if len(scenarios) >= 2
        }
        
        if not multi_scenario_pairs:
            logger.info(f"\n  ℹ️  No OD pairs found in multiple scenarios for {od_type} OD")
            return
        
        # Filter to only keep OD pairs where at least one model takes different routes
        varied_pairs = {}
        filtered_count = 0
        for od_pair, scenarios_dict in multi_scenario_pairs.items():
            if self._has_route_variation(scenarios_dict):
                varied_pairs[od_pair] = scenarios_dict
            else:
                filtered_count += 1
        
        if filtered_count > 0:
            logger.info(f"\n  ℹ️  Filtered out {filtered_count} OD pairs with identical routes across scenarios")
        
        if not varied_pairs:
            logger.info(f"  ℹ️  No OD pairs with route variation across scenarios for {od_type} OD")
            return
        
        logger.info(f"\n🔄 Generating multi-scenario comparisons for {len(varied_pairs)} OD pairs with route variation...")
        
        output_dir = self.config.output_dir / "scenario_cross_model" / od_type / "multi_scenario"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for od_pair, scenarios_dict in varied_pairs.items():
            origin, destination = od_pair
            scenario_names = sorted(scenarios_dict.keys())
            
            logger.info(f"  🔄 Origin {origin} → Dest {destination}: {len(scenario_names)} scenarios")
            
            # Create multi-panel figure (one column per scenario)
            n_scenarios = len(scenario_names)
            fig_width = 6 * n_scenarios
            fig_height = 6
            
            fig, axes = plt.subplots(1, n_scenarios, figsize=(fig_width, fig_height))
            if n_scenarios == 1:
                axes = [axes]
            
            # Plot each scenario in its own panel
            for idx, scenario in enumerate(scenario_names):
                ax = axes[idx]
                models_trajs = scenarios_dict[scenario]
                
                # Plot all models on this axis
                self._plot_multi_scenario_panel(
                    ax, models_trajs, scenario, origin, destination
                )
            
            # Overall title
            fig.suptitle(
                f"{od_type.upper()} OD: Origin {origin} → Destination {destination}\n"
                f"Across {n_scenarios} Scenarios: {', '.join([s.replace('_', ' ').title() for s in scenario_names])}",
                fontsize=14, fontweight='bold', y=0.98
            )
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save
            output_path = output_dir / f"{od_type}_origin{origin}_dest{destination}_multi_scenario"
            plt.savefig(f"{output_path}.pdf", dpi=self.config.dpi, bbox_inches='tight')
            plt.savefig(f"{output_path}.png", dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            logger.info(f"    ✅ Saved: {output_path}.{{pdf,png}}")
    
    def _plot_multi_scenario_panel(self, ax, models_trajs: Dict, scenario: str, 
                                   origin: int, destination: int):
        """Plot all models for one scenario on a subplot panel"""
        
        # Collect all coordinates for bounds
        all_lons, all_lats = [], []
        
        # Plot each model's trajectory
        for model_name in sorted(models_trajs.keys(), key=lambda x: (x != 'real', x)):
            traj = models_trajs[model_name]
            if not traj.coords:
                continue
            
            lons = [c[0] for c in traj.coords]
            lats = [c[1] for c in traj.coords]
            
            all_lons.extend(lons)
            all_lats.extend(lats)
            
            # Get plot styling
            color = self.comparison_plotter.model_colors.get(model_name, '#95a5a6')
            linestyle = self.comparison_plotter.model_linestyles.get(model_name, '-')
            label = self.comparison_plotter.model_labels.get(model_name, model_name)
            linewidth = 2.5 if model_name == 'real' else 2
            
            # Plot trajectory
            ax.plot(lons, lats,
                   color=color,
                   linestyle=linestyle,
                   linewidth=linewidth,
                   alpha=0.8,
                   label=label,
                   zorder=10 if model_name == 'real' else 5)
            
            # Mark start and end
            ax.scatter(lons[0], lats[0], c=color, marker='o', s=100,
                      zorder=15, edgecolors='white', linewidths=1.5)
            ax.scatter(lons[-1], lats[-1], c=color, marker='s', s=100,
                      zorder=15, edgecolors='white', linewidths=1.5)
        
        # Set bounds with margin
        if all_lons and all_lats:
            margin = self.config.margin
            ax.set_xlim(min(all_lons) - margin, max(all_lons) + margin)
            ax.set_ylim(min(all_lats) - margin, max(all_lats) + margin)
        
        # Styling
        ax.set_title(scenario.replace('_', ' ').title(), fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Longitude', fontsize=10)
        ax.set_ylabel('Latitude', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
        
        # Legend
        ax.legend(fontsize=9, loc='best', framealpha=0.95, edgecolor='black', fancybox=False)
    
    def _generate_scenario_cross_model_comparisons(self, gene_files: List[Dict]):
        """Generate cross-model comparisons per scenario (reuses OD matching logic)"""
        
        if not self.scenario_results or 'trajectory_mapping' not in self.scenario_results:
            logger.warning("No scenario mapping available for scenario cross-model mode")
            return
        
        logger.info("\n🎯 Generating scenario-based cross-model comparisons...")
        
        trajectory_mapping = self.scenario_results['trajectory_mapping']['trajectories']
        
        # For each OD type (train/test)
        for od_type in ['train', 'test']:
            logger.info(f"\n📊 Processing {od_type.upper()} OD scenarios...")
            
            # Load all models once
            models_data = self._load_all_models_for_od(gene_files, od_type)
            
            if len(models_data) < 2:
                logger.warning(f"⚠️  Not enough models loaded for {od_type} OD")
                continue
            
            # Get all scenarios (no filtering)
            scenarios = list(self.scenario_results['overview']['scenario_distribution'].keys())
            logger.info(f"  Found {len(scenarios)} scenarios")
            
            # Track OD pairs and their scenarios for multi-scenario plots
            od_scenario_map = {}  # {(origin, dest): {scenario: models_data}}
            
            # For each scenario: filter trajectories and call existing OD matching
            for scenario in scenarios:
                logger.info(f"  📍 Scenario: {scenario}")
                
                # Filter trajectories by scenario for each model
                scenario_models_data = {}
                for model_name, all_trajectories in models_data.items():
                    indices = [
                        t['index'] for t in trajectory_mapping
                        if scenario in t['scenarios'] and t['index'] < len(all_trajectories)
                    ]
                    
                    if indices:
                        scenario_models_data[model_name] = [all_trajectories[i] for i in indices]
                        logger.info(f"    {model_name}: {len(scenario_models_data[model_name])} trajectories")
                
                if len(scenario_models_data) < 2:
                    logger.warning(f"    Skipping {scenario} - not enough models with data")
                    continue
                
                # Track OD pairs in this scenario for multi-scenario plots
                self._track_od_pairs_for_multi_scenario(
                    scenario_models_data, scenario, od_scenario_map
                )
                
                # Call existing cross-model logic with scenario parameter
                self._plot_matching_od_pairs(scenario_models_data, od_type, scenario=scenario)
            
            # Generate multi-scenario comparison plots for OD pairs appearing in 2+ scenarios
            self._generate_multi_scenario_comparisons(od_scenario_map, od_type)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Main entry point with CLI interface"""
    parser = argparse.ArgumentParser(
        description="Trajectory Visualization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize trajectories from an evaluation directory
  uv run python visualize_trajectories.py --eval-dir hoser-distill-optuna-6
  
  # Use scenario-based sampling
  uv run python visualize_trajectories.py --eval-dir eval_xyz --sample_strategy scenario
  
  # Generate cross-model comparisons
  uv run python visualize_trajectories.py --eval-dir eval_xyz --cross_model
  
  # Only separate plots (no overlaid)
  uv run python visualize_trajectories.py --eval-dir eval_xyz --no_overlaid
        """
    )
    
    parser.add_argument('--eval-dir', required=True,
                        help='Evaluation directory containing results')
    parser.add_argument('--dataset', help='Dataset name (auto-detected from evaluation.yaml if not provided)')
    parser.add_argument('--sample_strategy', type=str, default='length_based',
                        choices=['random', 'length_based', 'representative', 'scenario'],
                        help='Sampling strategy for trajectory selection')
    parser.add_argument('--samples_per_type', type=int, default=1,
                        help='Number of samples per trajectory type (for random strategy)')
    parser.add_argument('--max_scenarios', type=int, default=5,
                        help='Maximum number of scenarios to plot (for scenario strategy)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no_separate', action='store_true',
                        help='Skip generating separate plots')
    parser.add_argument('--no_overlaid', action='store_true',
                        help='Skip generating overlaid plots')
    parser.add_argument('--cross_model', action='store_true',
                        help='Generate cross-model comparisons for same OD pairs')
    parser.add_argument('--no_real', action='store_true',
                        help='Exclude real trajectories from cross-model comparison (compare generated models only)')
    parser.add_argument('--scenario_cross_model', action='store_true',
                        help='Generate scenario-based cross-model comparisons (OD-matched within scenarios)')
    parser.add_argument('--basemap_style', type=str, default='none',
                        choices=['osm', 'gaode', 'cartodb', 'none'],
                        help='Basemap style: gaode (China-friendly), cartodb, osm, none (default: none)')
    parser.add_argument('--basemap_timeout', type=int, default=5,
                        help='Timeout for basemap requests in seconds (default: 5)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output resolution (DPI)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = VisualizationConfig(
        eval_dir=args.eval_dir,
        dataset=args.dataset,
        sample_strategy=args.sample_strategy,
        samples_per_type=args.samples_per_type,
        max_scenarios_to_plot=args.max_scenarios,
        random_seed=args.random_seed,
        generate_separate=not args.no_separate,
        generate_overlaid=not args.no_overlaid,
        generate_cross_model=args.cross_model,
        include_real_in_cross_model=not args.no_real,
        generate_scenario_cross_model=args.scenario_cross_model,
        basemap_style=args.basemap_style,
        basemap_timeout=args.basemap_timeout,
        dpi=args.dpi,
    )
    
    # Run visualizer
    visualizer = TrajectoryVisualizer(config)
    visualizer.run()


if __name__ == "__main__":
    main()

