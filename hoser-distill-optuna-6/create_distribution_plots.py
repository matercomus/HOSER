#!/usr/bin/env python3
"""
Create detailed distribution comparison plots for HOSER evaluation.

This script generates histograms and distribution plots comparing:
- Real train data vs generated (for each model)
- Real test data vs generated (for each model)
- Distance, duration, and radius of gyration distributions

Usage:
    uv run python create_distribution_plots.py
"""

import ast
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Set publication-quality defaults
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "font.family": "sans-serif",
    }
)

# Color scheme
COLORS = {
    "real_train": "#34495e",  # Dark gray
    "real_test": "#7f8c8d",  # Medium gray
    "distilled": "#27ae60",  # Green
    "distilled_seed44": "#16a085",  # Teal
    "vanilla": "#e74c3c",  # Red
}


class DistributionPlotter:
    """Generate distribution comparison plots"""

    def __init__(
        self,
        data_dir: str = "../data/Beijing",
        gene_dir: str = "gene/Beijing/seed42",
        eval_dir: str = "eval",
        output_dir: str = "figures/distributions",
    ):
        self.data_dir = Path(data_dir)
        self.gene_dir = Path(gene_dir)
        self.eval_dir = Path(eval_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load road network for coordinate conversion
        self.road_coords = self._load_road_network()

    def _load_road_network(self) -> Dict[int, Tuple[float, float]]:
        """Load road network centroids"""
        logger.info("📂 Loading road network...")
        roadmap_path = self.data_dir / "roadmap.geo"

        df = pl.read_csv(
            roadmap_path,
            has_header=True,
            schema_overrides={"lanes": pl.Utf8, "oneway": pl.Utf8},
        )

        road_coords = {}
        for row in df.iter_rows(named=True):
            road_id = row["geo_id"]
            coordinates_str = row["coordinates"]

            try:
                coords = ast.literal_eval(coordinates_str)
                # Calculate centroid
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                centroid = (np.mean(lons), np.mean(lats))
                road_coords[road_id] = centroid
            except (ValueError, SyntaxError):
                continue

        logger.info(f"✅ Loaded {len(road_coords)} roads")
        return road_coords

    def _calculate_haversine(
        self, lon1: float, lat1: float, lon2: float, lat2: float
    ) -> float:
        """Calculate Haversine distance in kilometers"""
        R = 6371  # Earth radius in km

        # Convert to radians
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def _calculate_trajectory_distance(self, road_ids: List[int]) -> float:
        """Calculate total trajectory distance"""
        if len(road_ids) < 2:
            return 0.0

        total_dist = 0.0
        for i in range(len(road_ids) - 1):
            rid1, rid2 = road_ids[i], road_ids[i + 1]
            if rid1 in self.road_coords and rid2 in self.road_coords:
                lon1, lat1 = self.road_coords[rid1]
                lon2, lat2 = self.road_coords[rid2]
                total_dist += self._calculate_haversine(lon1, lat1, lon2, lat2)

        return total_dist

    def _calculate_radius_of_gyration(self, road_ids: List[int]) -> float:
        """Calculate radius of gyration"""
        if len(road_ids) < 2:
            return 0.0

        # Get all coordinates
        coords = [self.road_coords[rid] for rid in road_ids if rid in self.road_coords]
        if len(coords) < 2:
            return 0.0

        # Calculate centroid
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        center_lon, center_lat = np.mean(lons), np.mean(lats)

        # Calculate radius of gyration (RMS distance from centroid)
        squared_dists = []
        for lon, lat in coords:
            dist = self._calculate_haversine(lon, lat, center_lon, center_lat)
            squared_dists.append(dist**2)

        return np.sqrt(np.mean(squared_dists))

    def _load_real_data(
        self, csv_path: Path, sample_size: int = 50000
    ) -> Dict[str, List[float]]:
        """Load and compute metrics from real data"""
        logger.info(
            f"📂 Loading real data from {csv_path.name} (sample: {sample_size})..."
        )

        df = pl.read_csv(csv_path).head(sample_size)

        distances = []
        radii = []

        for row in df.iter_rows(named=True):
            try:
                road_ids = ast.literal_eval(row["rid_list"])

                # Calculate metrics
                dist = self._calculate_trajectory_distance(road_ids)
                radius = self._calculate_radius_of_gyration(road_ids)

                if dist > 0:
                    distances.append(dist)
                if radius > 0:
                    radii.append(radius)
            except (ValueError, SyntaxError, KeyError):
                continue

        logger.info(f"✅ Computed metrics for {len(distances)} real trajectories")
        return {
            "distances": distances,
            "radii": radii,
        }

    def _load_generated_data(self, csv_path: Path) -> Dict[str, List[float]]:
        """Load and compute metrics from generated data"""
        logger.info(f"📂 Loading generated data from {csv_path.name}...")

        df = pl.read_csv(csv_path)

        distances = []
        radii = []

        for row in df.iter_rows(named=True):
            try:
                road_ids = ast.literal_eval(row["gene_trace_road_id"])

                # Calculate metrics
                dist = self._calculate_trajectory_distance(road_ids)
                radius = self._calculate_radius_of_gyration(road_ids)

                if dist > 0:
                    distances.append(dist)
                if radius > 0:
                    radii.append(radius)
            except (ValueError, SyntaxError, KeyError):
                continue

        logger.info(f"✅ Computed metrics for {len(distances)} generated trajectories")
        return {
            "distances": distances,
            "radii": radii,
        }

    def plot_distance_distributions(self):
        """Plot distance distribution comparisons"""
        logger.info("📊 Creating distance distribution plots...")

        # Load real data
        real_train = self._load_real_data(self.data_dir / "train.csv")
        real_test = self._load_real_data(self.data_dir / "test.csv")

        # Load generated data
        generated_data = {}
        for csv_file in sorted(self.gene_dir.glob("*.csv")):
            if "vanilla_train" in csv_file.name:
                generated_data["vanilla_train"] = self._load_generated_data(csv_file)
            elif "vanilla_test" in csv_file.name:
                generated_data["vanilla_test"] = self._load_generated_data(csv_file)
            elif "distilled_seed44_train" in csv_file.name:
                generated_data["distilled_seed44_train"] = self._load_generated_data(
                    csv_file
                )
            elif "distilled_seed44_test" in csv_file.name:
                generated_data["distilled_seed44_test"] = self._load_generated_data(
                    csv_file
                )
            elif "distilled_train" in csv_file.name:
                generated_data["distilled_train"] = self._load_generated_data(csv_file)
            elif "distilled_test" in csv_file.name:
                generated_data["distilled_test"] = self._load_generated_data(csv_file)

        # Create plots for train OD
        self._plot_distance_comparison(
            real_train["distances"],
            {k: v["distances"] for k, v in generated_data.items() if "train" in k},
            "Train OD: Distance Distribution Comparison",
            self.output_dir / "distance_distribution_train_od",
        )

        # Create plots for test OD
        self._plot_distance_comparison(
            real_test["distances"],
            {k: v["distances"] for k, v in generated_data.items() if "test" in k},
            "Test OD: Distance Distribution Comparison",
            self.output_dir / "distance_distribution_test_od",
        )

        # Create radius plots
        self._plot_radius_comparison(
            real_train["radii"],
            {k: v["radii"] for k, v in generated_data.items() if "train" in k},
            "Train OD: Radius of Gyration Distribution",
            self.output_dir / "radius_distribution_train_od",
        )

        self._plot_radius_comparison(
            real_test["radii"],
            {k: v["radii"] for k, v in generated_data.items() if "test" in k},
            "Test OD: Radius of Gyration Distribution",
            self.output_dir / "radius_distribution_test_od",
        )

    def _plot_distance_comparison(
        self,
        real_distances: List[float],
        generated_distances: Dict[str, List[float]],
        title: str,
        output_path: Path,
    ):
        """Create distance distribution comparison plot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Density curves (KDE)
        from scipy import stats

        x_range = np.linspace(0, 30, 300)  # Smooth curve with 300 points

        # Plot real data with thicker line
        kde_real = stats.gaussian_kde(real_distances)
        ax1.plot(
            x_range,
            kde_real(x_range),
            color=COLORS["real_train"],
            linewidth=3,
            label="Real",
            linestyle="-",
            alpha=0.9,
        )

        # Plot generated models with distinct styles
        linestyles = {"distilled": "-", "distilled_seed44": "--", "vanilla": "-."}
        for model_key, distances in sorted(generated_distances.items()):
            model_name = model_key.replace("_train", "").replace("_test", "")
            color = COLORS.get(model_name, "#95a5a6")
            linestyle = linestyles.get(model_name, "-")

            kde = stats.gaussian_kde(distances)
            ax1.plot(
                x_range,
                kde(x_range),
                color=color,
                linewidth=2.5,
                label=model_name.title(),
                linestyle=linestyle,
                alpha=0.85,
            )

        ax1.set_xlabel("Distance (km)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Density", fontsize=12, fontweight="bold")
        ax1.set_title("Distance Distributions", fontsize=13, fontweight="bold")
        ax1.legend(fontsize=11, framealpha=0.95)
        ax1.grid(True, alpha=0.3, linestyle="--")

        # Right: Box plots
        box_data = [real_distances]
        box_labels = ["Real"]
        box_colors = [COLORS["real_train"]]

        for model_key, distances in sorted(generated_distances.items()):
            model_name = model_key.replace("_train", "").replace("_test", "")
            box_data.append(distances)
            box_labels.append(model_name.title())
            box_colors.append(COLORS.get(model_name, "#95a5a6"))

        bp = ax2.boxplot(
            box_data, tick_labels=box_labels, patch_artist=True, showfliers=False
        )

        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.set_ylabel("Distance (km)", fontsize=12, fontweight="bold")
        ax2.set_title("Distribution Spread", fontsize=13, fontweight="bold")
        ax2.set_xticklabels(box_labels, rotation=15, ha="right", fontsize=10)
        ax2.grid(True, alpha=0.3, axis="y", linestyle="--")

        # Add mean as diamonds
        for i, (data, color) in enumerate(zip(box_data, box_colors), start=1):
            mean_val = np.mean(data)
            ax2.plot(
                i,
                mean_val,
                marker="D",
                color=color,
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=1,
                zorder=3,
            )

        fig.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()

        # Save both formats
        plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"✅ Saved: {output_path}.{{pdf,png}}")

    def _plot_radius_comparison(
        self,
        real_radii: List[float],
        generated_radii: Dict[str, List[float]],
        title: str,
        output_path: Path,
    ):
        """Create radius of gyration comparison plot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left: Density curves (KDE)
        from scipy import stats

        x_range = np.linspace(0, 5, 300)  # Smooth curve with 300 points

        # Plot real data with thicker line
        kde_real = stats.gaussian_kde(real_radii)
        ax1.plot(
            x_range,
            kde_real(x_range),
            color=COLORS["real_train"],
            linewidth=3,
            label="Real",
            linestyle="-",
            alpha=0.9,
        )

        # Plot generated models with distinct styles
        linestyles = {"distilled": "-", "distilled_seed44": "--", "vanilla": "-."}
        for model_key, radii in sorted(generated_radii.items()):
            model_name = model_key.replace("_train", "").replace("_test", "")
            color = COLORS.get(model_name, "#95a5a6")
            linestyle = linestyles.get(model_name, "-")

            kde = stats.gaussian_kde(radii)
            ax1.plot(
                x_range,
                kde(x_range),
                color=color,
                linewidth=2.5,
                label=model_name.title(),
                linestyle=linestyle,
                alpha=0.85,
            )

        ax1.set_xlabel("Radius of Gyration (km)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Density", fontsize=12, fontweight="bold")
        ax1.set_title(
            "Radius of Gyration Distributions", fontsize=13, fontweight="bold"
        )
        ax1.legend(fontsize=11, framealpha=0.95)
        ax1.grid(True, alpha=0.3, linestyle="--")

        # Right: Box plots
        box_data = [real_radii]
        box_labels = ["Real"]
        box_colors = [COLORS["real_train"]]

        for model_key, radii in generated_radii.items():
            model_name = model_key.replace("_train", "").replace("_test", "")
            box_data.append(radii)
            box_labels.append(model_name.title())
            box_colors.append(COLORS.get(model_name, "#95a5a6"))

        bp = ax2.boxplot(
            box_data, tick_labels=box_labels, patch_artist=True, showfliers=False
        )

        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax2.set_ylabel("Radius of Gyration (km)", fontsize=12, fontweight="bold")
        ax2.set_title("Distribution Spread", fontsize=13, fontweight="bold")
        ax2.set_xticklabels(box_labels, rotation=15, ha="right", fontsize=10)
        ax2.grid(True, alpha=0.3, axis="y", linestyle="--")

        # Add mean as diamonds
        for i, (data, color) in enumerate(zip(box_data, box_colors), start=1):
            mean_val = np.mean(data)
            ax2.plot(
                i,
                mean_val,
                marker="D",
                color=color,
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=1,
                zorder=3,
            )

        fig.suptitle(title, fontsize=16, fontweight="bold")
        plt.tight_layout()

        # Save both formats
        plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches="tight")
        plt.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"✅ Saved: {output_path}.{{pdf,png}}")

    def generate_all_plots(self):
        """Generate all distribution plots"""
        logger.info("🎨 Generating distribution plots...")
        self.plot_distance_distributions()
        logger.info("✅ All distribution plots generated!")


def main():
    """Main execution"""
    plotter = DistributionPlotter()
    plotter.generate_all_plots()


if __name__ == "__main__":
    main()
