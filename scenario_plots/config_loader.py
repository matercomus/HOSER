"""
Configuration loader for scenario plots.

Manages plot registry and configuration from YAML files.
"""

import fnmatch
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PlotConfig:
    """Configuration for a single plot"""

    id: str
    module_name: str
    functions: List[str]  # Can be single function or multiple
    description: str
    enabled: bool
    config: Dict[str, Any]  # Plot-specific settings


class PlotConfigLoader:
    """Loads and manages plot configuration from YAML registry"""

    def __init__(self, config_dir: Path):
        """Initialize loader with config directory

        Args:
            config_dir: Directory containing scenario_plots.yaml
        """
        self.config_dir = Path(config_dir)
        self.registry_file = self.config_dir / "scenario_plots.yaml"
        self.registry = self._load_plot_registry()

    def _load_plot_registry(self) -> Dict:
        """Load global plot registry from config/scenario_plots.yaml"""
        if not self.registry_file.exists():
            logger.warning(
                f"Plot registry not found: {self.registry_file}, using defaults"
            )
            return {}

        try:
            with open(self.registry_file) as f:
                registry = yaml.safe_load(f)
            logger.info(f"📋 Loaded plot registry from {self.registry_file.name}")
            return registry or {}
        except Exception as e:
            logger.error(f"Failed to load plot registry: {e}")
            return {}

    def get_enabled_plots(self, group_or_id: str = "all") -> List[PlotConfig]:
        """Get list of enabled plots for a group or individual plot

        Args:
            group_or_id: Group name (e.g., "all", "core") or plot ID (e.g., "metrics.scenario_heatmap")

        Returns:
            List of PlotConfig objects
        """
        if "." in group_or_id:
            # Individual plot ID
            plot = self.get_plot_by_id(group_or_id)
            return [plot] if plot else []
        else:
            # Group name
            return self._resolve_group(group_or_id)

    def get_plot_by_id(self, plot_id: str) -> Optional[PlotConfig]:
        """Get a specific plot by its ID

        Args:
            plot_id: Plot identifier like "application.improvement_heatmaps"

        Returns:
            PlotConfig object or None if not found
        """
        if "." not in plot_id:
            logger.warning(f"Invalid plot ID format: {plot_id} (expected type.plot)")
            return None

        plot_type, plot_name = plot_id.split(".", 1)

        plot_types = self.registry.get("plot_types", {})
        if plot_type not in plot_types:
            logger.warning(f"Plot type not found: {plot_type}")
            return None

        type_config = plot_types[plot_type]
        plots = type_config.get("plots", {})

        if plot_name not in plots:
            logger.warning(f"Plot not found: {plot_name} in {plot_type}")
            return None

        plot_data = plots[plot_name]
        module_name = type_config["module"]

        # Handle single function or multiple functions
        if "function" in plot_data:
            functions = [plot_data["function"]]
        elif "functions" in plot_data:
            functions = plot_data["functions"]
        else:
            logger.warning(f"No function(s) defined for plot: {plot_id}")
            return None

        return PlotConfig(
            id=plot_id,
            module_name=module_name,
            functions=functions,
            description=plot_data.get("description", ""),
            enabled=plot_data.get("enabled", True),
            config=plot_data.get("config", {}),
        )

    def _resolve_group(self, group_name: str) -> List[PlotConfig]:
        """Resolve group name to list of plot IDs

        Args:
            group_name: Name of group (e.g., "all", "core", "heatmaps_only")

        Returns:
            List of PlotConfig objects
        """
        groups = self.registry.get("groups", {})

        if group_name not in groups:
            logger.warning(f"Group not found: {group_name}")
            return []

        group_config = groups[group_name]
        plot_specs = group_config.get("plots", [])

        if plot_specs == "*":
            # All enabled plots
            return self._get_all_enabled_plots()

        # Resolve list of plot specs (can include wildcards)
        plots = []
        for spec in plot_specs:
            if "*" in spec:
                # Wildcard pattern like "metrics.*"
                plots.extend(self._expand_wildcard(spec))
            else:
                # Specific plot ID
                plot = self.get_plot_by_id(spec)
                if plot and plot.enabled:
                    plots.append(plot)

        return plots

    def _expand_wildcard(self, pattern: str) -> List[PlotConfig]:
        """Expand wildcard pattern to matching plots

        Args:
            pattern: Pattern like "metrics.*" or "application.*"

        Returns:
            List of matching PlotConfig objects
        """
        plots = []
        plot_types = self.registry.get("plot_types", {})

        for plot_type, type_config in plot_types.items():
            for plot_name in type_config.get("plots", {}).keys():
                plot_id = f"{plot_type}.{plot_name}"
                if fnmatch.fnmatch(plot_id, pattern):
                    plot = self.get_plot_by_id(plot_id)
                    if plot and plot.enabled:
                        plots.append(plot)

        return plots

    def _get_all_enabled_plots(self) -> List[PlotConfig]:
        """Get all enabled plots from registry

        Returns:
            List of all enabled PlotConfig objects
        """
        plots = []
        plot_types = self.registry.get("plot_types", {})

        for plot_type, type_config in plot_types.items():
            for plot_name in type_config.get("plots", {}).keys():
                plot_id = f"{plot_type}.{plot_name}"
                plot = self.get_plot_by_id(plot_id)
                if plot and plot.enabled:
                    plots.append(plot)

        return plots

    def list_available_groups(self) -> Dict[str, str]:
        """Get dictionary of available groups and their descriptions

        Returns:
            Dictionary mapping group names to descriptions
        """
        groups = self.registry.get("groups", {})
        return {name: group.get("description", "") for name, group in groups.items()}

    def list_available_plots(self) -> Dict[str, str]:
        """Get dictionary of available plots and their descriptions

        Returns:
            Dictionary mapping plot IDs to descriptions
        """
        plots = {}
        plot_types = self.registry.get("plot_types", {})

        for plot_type, type_config in plot_types.items():
            for plot_name, plot_data in type_config.get("plots", {}).items():
                plot_id = f"{plot_type}.{plot_name}"
                plots[plot_id] = plot_data.get("description", "")

        return plots

