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
import logging
import random
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
    # Paths
    roadmap_path: Path = Path("../data/Beijing/roadmap.geo")
    train_csv_path: Path = Path("../data/Beijing/train.csv")
    test_csv_path: Path = Path("../data/Beijing/test.csv")
    gene_dir: Path = Path("gene/Beijing/seed42")
    
    # Sampling strategy
    sample_strategy: str = "length_based"  # "random", "length_based", "representative"
    samples_per_type: int = 1
    random_seed: int = 42
    
    # Comparison modes
    generate_separate: bool = True
    generate_overlaid: bool = True
    generate_cross_model: bool = False  # Compare all models for same OD pair
    
    # Basemap
    basemap_style: str = "none"  # "osm", "gaode", "cartodb", "none" - default to none for speed
    basemap_timeout: int = 5  # seconds - fast timeout for China network
    basemap_test_first: bool = True  # Test connectivity before fetching all tiles
    
    # Output
    output_dir: Path = Path("figures/trajectories")
    dpi: int = 300
    figsize: Tuple[int, int] = (12, 10)
    
    # Visualization
    margin: float = 0.002  # Map padding in degrees (~200m)
    
    def __post_init__(self):
        """Convert string paths to Path objects"""
        self.roadmap_path = Path(self.roadmap_path)
        self.train_csv_path = Path(self.train_csv_path)
        self.test_csv_path = Path(self.test_csv_path)
        self.gene_dir = Path(self.gene_dir)
        self.output_dir = Path(self.output_dir)


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
            
            if len(trajectories) >= sample_size:
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
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
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
# Trajectory Plotter
# =============================================================================

class TrajectoryPlotter:
    """Plot trajectories on maps with various styles"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.basemap_manager = BasemapManager(config)
    
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
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Convert coordinates for basemap if needed (WGS84 -> GCJ-02 for Gaode)
        plot_coords = self._convert_coords_for_basemap(trajectory.coords)
        lons, lats = zip(*plot_coords)
        
        # Plot trajectory line
        ax.plot(lons, lats, 'b-', linewidth=2.5, label='Trajectory', zorder=3, alpha=0.8)
        
        # Start marker (green circle)
        ax.scatter(lons[0], lats[0], c='green', s=150, marker='o', 
                   label='Start', zorder=4, edgecolors='black', linewidths=2)
        
        # End marker (red square)
        ax.scatter(lons[-1], lats[-1], c='red', s=150, marker='s', 
                   label='End', zorder=4, edgecolors='black', linewidths=2)
        
        # Auto-zoom with padding
        margin = self.config.margin
        ax.set_xlim(min(lons) - margin, max(lons) + margin)
        ax.set_ylim(min(lats) - margin, max(lats) + margin)
        
        # Add basemap if configured
        if self.config.basemap_style != "none" and cx is not None:
            self.basemap_manager.add_basemap_safe(ax, crs='EPSG:4326', zoom='auto')
        
        # Styling
        if title is None:
            title = f"{trajectory.model} - {trajectory.od_type} OD - {trajectory.source}"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.grid(True, alpha=0.3)
        
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
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
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
        
        # Auto-zoom with padding
        if all_lons and all_lats:
            margin = self.config.margin
            ax.set_xlim(min(all_lons) - margin, max(all_lons) + margin)
            ax.set_ylim(min(all_lats) - margin, max(all_lats) + margin)
        
        # Add basemap if configured
        if self.config.basemap_style != "none" and cx is not None:
            self.basemap_manager.add_basemap_safe(ax, crs='EPSG:4326', zoom='auto')
        
        # Styling
        if title is None:
            first_traj = next(iter(trajectories.values()))
            title = f"{first_traj.model} - {first_traj.od_type} OD - All Lengths"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Save both formats
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{output_path}.pdf", dpi=self.config.dpi, bbox_inches='tight')
        plt.savefig(f"{output_path}.png", dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Saved: {output_path}.{{pdf,png}}")
    
    def plot_cross_model_comparison(self, trajectories: Dict[str, Trajectory], output_path: Path, 
                                   title: str = None, missing_models: List[str] = None):
        """Plot trajectories from different models for the same OD pair"""
        if not trajectories:
            logger.warning("No trajectories to plot")
            return
        
        if missing_models is None:
            missing_models = []
        
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
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
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
            alpha = 0.9 if model_name == 'real' else 0.7
            
            # Plot line
            ax.plot(lons, lats, linestyle=linestyle, linewidth=linewidth, 
                   label=label, color=color, zorder=4 if model_name == 'real' else 3, alpha=alpha)
            
            # Start marker
            ax.scatter(lons[0], lats[0], c=color, s=120, marker='o', 
                      zorder=5, edgecolors='black', linewidths=1.5, alpha=0.9)
            
            # End marker
            ax.scatter(lons[-1], lats[-1], c=color, s=120, marker='s', 
                      zorder=5, edgecolors='black', linewidths=1.5, alpha=0.9)
        
        # Auto-zoom with padding
        if all_lons and all_lats:
            margin = self.config.margin
            ax.set_xlim(min(all_lons) - margin, max(all_lons) + margin)
            ax.set_ylim(min(all_lats) - margin, max(all_lats) + margin)
        
        # Add basemap if configured
        if self.config.basemap_style != "none" and cx is not None:
            self.basemap_manager.add_basemap_safe(ax, crs='EPSG:4326', zoom='auto')
        
        # Styling
        if title is None:
            title = "Cross-Model Comparison - Same OD Pair"
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        # Create custom legend with start/end markers
        from matplotlib.lines import Line2D
        
        legend_elements = []
        
        # Add model trajectories to legend
        for model_name in sorted(trajectories.keys(), key=lambda x: (x != 'real', x)):
            color = model_colors.get(model_name, '#333333')
            linestyle = model_linestyles.get(model_name, '-')
            label = model_labels.get(model_name, model_name)
            linewidth = 3.5 if model_name == 'real' else 2.5
            
            legend_elements.append(Line2D([0], [0], color=color, linewidth=linewidth, 
                                         linestyle=linestyle, label=label))
        
        # Add separator
        legend_elements.append(Line2D([0], [0], color='none', label=''))
        
        # Add start/end markers
        legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor='gray', markersize=10, 
                                     markeredgecolor='black', markeredgewidth=1.5,
                                     label='Start Point', linestyle=''))
        legend_elements.append(Line2D([0], [0], marker='s', color='w', 
                                     markerfacecolor='gray', markersize=10,
                                     markeredgecolor='black', markeredgewidth=1.5,
                                     label='End Point', linestyle=''))
        
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11, 
                 framealpha=0.95, title='Legend', title_fontsize=12)
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.grid(True, alpha=0.3)
        
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
        
        # Initialize components
        self.loader = TrajectoryLoader(config, self.road_network)
        self.sampler = TrajectorySampler(config)
        self.plotter = TrajectoryPlotter(config)
        
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
        
        logger.info("✅ Visualization pipeline completed!")
    
    def _detect_gene_files(self) -> List[Dict]:
        """Detect generated trajectory CSV files"""
        logger.info(f"🔍 Detecting generated trajectory files in {self.config.gene_dir}")
        
        gene_files = []
        
        for csv_file in self.config.gene_dir.glob("*.csv"):
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
            
            # Also load real trajectories (increased sample size to find more common OD pairs)
            logger.info(f"  Loading real {od_type} trajectories...")
            real_csv = self.config.train_csv_path if od_type == 'train' else self.config.test_csv_path
            real_trajectories = self.loader.load_real_trajectories(real_csv, od_type, sample_size=2000)
            
            if real_trajectories:
                models_data['real'] = real_trajectories
            
            if len(models_data) < 2:
                logger.warning(f"⚠️  Not enough models loaded for {od_type} OD comparison")
                continue
            
            # Find common OD pairs
            self._plot_matching_od_pairs(models_data, od_type)
    
    def _plot_matching_od_pairs(self, models_data: Dict[str, List[Trajectory]], od_type: str):
        """Find and plot trajectories with matching OD pairs across models"""
        
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
        
        # Sample a few representative OD pairs
        sampled_od_pairs = self._sample_od_pairs_for_comparison(list(all_od_pairs), od_indices, models_data)
        
        # Generate comparison plots
        output_dir = self.config.output_dir / "cross_model"
        
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
            title = f"{od_type.upper()} OD: All Models - Origin {origin} → Destination {destination}"
            
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
        
        # Sample multiple representative lengths (more examples)
        n = len(sorted_od_pairs)
        if n >= 6:
            # Sample 6 trajectories: very short, short, medium-short, medium, medium-long, long
            percentiles = [0.10, 0.25, 0.40, 0.60, 0.75, 0.90]
            indices = [int(n * p) for p in percentiles]
            return [sorted_od_pairs[i][0] for i in indices]
        elif n >= 3:
            # Fallback to short, medium, long
            indices = [int(n * 0.25), int(n * 0.50), int(n * 0.75)]
            return [sorted_od_pairs[i][0] for i in indices]
        else:
            # Return all available
            return [pair for pair, _ in sorted_od_pairs]


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
  # Length-based sampling (default: short, medium, long)
  uv run python visualize_trajectories.py --sample_strategy length_based
  
  # Random sampling
  uv run python visualize_trajectories.py --sample_strategy random --samples_per_type 3
  
  # Representative (median) sampling
  uv run python visualize_trajectories.py --sample_strategy representative
  
  # Only separate plots (no overlaid)
  uv run python visualize_trajectories.py --no_overlaid
  
  # Only overlaid plots (no separate)
  uv run python visualize_trajectories.py --no_separate
        """
    )
    
    parser.add_argument('--sample_strategy', type=str, default='length_based',
                        choices=['random', 'length_based', 'representative'],
                        help='Sampling strategy for trajectory selection')
    parser.add_argument('--samples_per_type', type=int, default=1,
                        help='Number of samples per trajectory type (for random strategy)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no_separate', action='store_true',
                        help='Skip generating separate plots')
    parser.add_argument('--no_overlaid', action='store_true',
                        help='Skip generating overlaid plots')
    parser.add_argument('--cross_model', action='store_true',
                        help='Generate cross-model comparisons for same OD pairs (includes real trajectories)')
    parser.add_argument('--basemap_style', type=str, default='none',
                        choices=['osm', 'gaode', 'cartodb', 'none'],
                        help='Basemap style: gaode (China-friendly), cartodb, osm, none (default: none)')
    parser.add_argument('--basemap_timeout', type=int, default=5,
                        help='Timeout for basemap requests in seconds (default: 5)')
    parser.add_argument('--output_dir', type=str, default='figures/trajectories',
                        help='Output directory for figures')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Output resolution (DPI)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = VisualizationConfig(
        sample_strategy=args.sample_strategy,
        samples_per_type=args.samples_per_type,
        random_seed=args.random_seed,
        generate_separate=not args.no_separate,
        generate_overlaid=not args.no_overlaid,
        generate_cross_model=args.cross_model,
        basemap_style=args.basemap_style,
        basemap_timeout=args.basemap_timeout,
        output_dir=Path(args.output_dir),
        dpi=args.dpi,
    )
    
    # Run visualizer
    visualizer = TrajectoryVisualizer(config)
    visualizer.run()


if __name__ == "__main__":
    main()

