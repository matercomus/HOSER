"""
Data loading utilities for scenario analysis.
"""

import fnmatch
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import seaborn as sns
import yaml

logger = logging.getLogger(__name__)


# Consistent color palette for all model visualizations
MODEL_COLOR_PALETTE = "husl"


class ScenarioDataLoader:
    """Load and manage scenario analysis data"""

    def __init__(self, eval_dir: Path):
        self.eval_dir = Path(eval_dir).resolve()
        self.scenarios_dir = self.eval_dir / "scenarios"

        if not self.scenarios_dir.exists():
            raise FileNotFoundError(
                f"Scenarios directory not found: {self.scenarios_dir}"
            )

        # Load dataset-specific configuration
        self.config_file = self._find_scenario_config()
        self.dataset_config = self._load_dataset_config()

    def _find_scenario_config(self) -> Optional[Path]:
        """Find scenarios_*.yaml in eval_dir/config/"""
        config_dir = self.eval_dir / "config"
        if not config_dir.exists():
            logger.warning(f"Config directory not found: {config_dir}")
            return None

        # Look for scenarios_*.yaml files
        config_files = list(config_dir.glob("scenarios_*.yaml"))

        if not config_files:
            logger.warning(f"No scenarios_*.yaml found in {config_dir}")
            return None

        if len(config_files) > 1:
            logger.warning(
                f"Multiple scenario configs found, using first: {config_files[0].name}"
            )

        return config_files[0]

    def _load_dataset_config(self) -> Dict:
        """Load dataset-specific configuration from YAML"""
        if not self.config_file or not self.config_file.exists():
            logger.info("No dataset config found, using default settings")
            return {}

        try:
            with open(self.config_file) as f:
                config = yaml.safe_load(f)
            logger.info(f"📋 Loaded config from {self.config_file.name}")
            return config or {}
        except Exception as e:
            logger.warning(f"Failed to load config from {self.config_file}: {e}")
            return {}

    def get_filtered_metrics(self, all_metrics: List[str]) -> List[str]:
        """Apply plotting.metrics filters to metric list

        Args:
            all_metrics: List of all available metrics

        Returns:
            Filtered list of metrics based on config rules
        """
        config = self.dataset_config.get("plotting", {}).get("metrics", {})

        if not config:
            # No filtering configured
            return all_metrics

        filtered = list(all_metrics)

        # Apply exclude_patterns (use fnmatch for wildcards)
        exclude_patterns = config.get("exclude_patterns", [])
        for pattern in exclude_patterns:
            filtered = [m for m in filtered if not fnmatch.fnmatch(m, pattern)]

        # Apply exact excludes
        excludes = set(config.get("exclude", []))
        filtered = [m for m in filtered if m not in excludes]

        # Apply include_only if specified (overrides excludes)
        include_only = config.get("include_only")
        if include_only:
            filtered = [m for m in filtered if m in include_only]

        # Log filtering results
        if len(filtered) < len(all_metrics):
            excluded_count = len(all_metrics) - len(filtered)
            logger.info(
                f"🎯 Filtered {excluded_count} metrics ({len(filtered)} remaining)"
            )

        return filtered

    def get_plot_override(self, plot_id: str) -> Dict:
        """Get dataset-specific overrides for a plot

        Args:
            plot_id: Plot identifier (e.g., "application.improvement_heatmaps")

        Returns:
            Dictionary of config overrides for this plot
        """
        overrides = self.dataset_config.get("plotting", {}).get("plot_overrides", {})
        return overrides.get(plot_id, {})

    def load_all(self) -> Dict:
        """Load all scenario data from JSON files"""
        logger.info(f"📂 Loading scenario data from {self.eval_dir}")

        data = {}

        # Auto-detect OD sources
        for od_dir in self.scenarios_dir.iterdir():
            if od_dir.is_dir() and not od_dir.name.startswith("."):
                od_source = od_dir.name
                data[od_source] = {}

                # Load each model
                for model_dir in od_dir.iterdir():
                    if model_dir.is_dir():
                        model = model_dir.name
                        metrics_file = model_dir / "scenario_metrics.json"

                        if metrics_file.exists():
                            with open(metrics_file) as f:
                                data[od_source][model] = json.load(f)
                            logger.info(f"  ✅ {od_source}/{model}")

        if not data:
            raise ValueError("No scenario data found!")

        return data

    def get_common_scenarios(self, data: Dict, od_source: str = "train") -> List[str]:
        """Get scenarios that exist across all models"""
        if od_source not in data:
            return []

        # Get scenarios from first model
        first_model = list(data[od_source].keys())[0]
        scenarios = set(
            data[od_source][first_model].get("individual_scenarios", {}).keys()
        )

        # Intersect with other models
        for model in data[od_source].keys():
            model_scenarios = set(
                data[od_source][model].get("individual_scenarios", {}).keys()
            )
            scenarios &= model_scenarios

        # Return in consistent order
        return sorted(list(scenarios))

    def get_models(self, data: Dict, od_source: str = "train") -> List[str]:
        """Get list of models for a given OD source"""
        if od_source not in data:
            return []
        return sorted(data[od_source].keys())


def get_scenario_list(
    data: Dict, od_source: str = "train", model: str = "vanilla"
) -> List[str]:
    """Extract ordered list of scenarios"""
    if od_source not in data or model not in data[od_source]:
        return []

    scenarios = data[od_source][model].get("individual_scenarios", {}).keys()
    return sorted(list(scenarios))


def get_metric_value(
    data: Dict, od_source: str, model: str, scenario: str, metric: str
) -> Optional[float]:
    """Safe metric extraction"""
    try:
        return data[od_source][model]["individual_scenarios"][scenario]["metrics"][
            metric
        ]
    except KeyError:
        logger.warning(f"Metric not found: {od_source}/{model}/{scenario}/{metric}")
        return None


def calculate_model_quality(
    data: Dict,
    od_source: str,
    scenario: str,
    metric: str,
    model: str,
    reference_max: Optional[float] = None,
) -> Optional[float]:
    """Calculate model quality score based on distance from real data.
    
    All metrics (JSD, Hausdorff, DTW, EDR) measure distance from reality where:
    - 0 = perfect match with real data
    - Higher values = worse match
    
    Quality score: 100 = perfect, 0 = matches reference_max, negative = worse than reference
    
    Args:
        data: Loaded scenario data
        od_source: 'train' or 'test'
        scenario: Scenario name
        metric: Metric name
        model: Model name to evaluate
        reference_max: Maximum expected value for normalization (uses 95th percentile if None)
    
    Returns:
        Quality score (100 = perfect, 0 = reference_max, can be negative)
    """
    metric_value = get_metric_value(data, od_source, model, scenario, metric)
    
    if metric_value is None:
        return None
    
    # If no reference provided, calculate from all models' values for this metric/scenario
    if reference_max is None:
        all_values = []
        for model_name in data[od_source].keys():
            val = get_metric_value(data, od_source, model_name, scenario, metric)
            if val is not None:
                all_values.append(val)
        
        if not all_values:
            return None
        
        # Use 95th percentile as reference maximum
        reference_max = np.percentile(all_values, 95)
        
        # Avoid division by zero
        if reference_max < 0.0001:
            reference_max = 0.01
    
    # Calculate quality: 100 at metric_value=0, 0 at metric_value=reference_max
    # Allow negative scores for values worse than reference_max
    quality = (1 - (metric_value / reference_max)) * 100
    
    return quality


def classify_models(
    data: Dict, od_source: str = "train"
) -> Tuple[List[str], List[str]]:
    """Classify models into vanilla and distilled lists

    Args:
        data: Loaded scenario data
        od_source: OD source to analyze (default: 'train')

    Returns:
        Tuple of (vanilla_models, distilled_models), both sorted alphabetically
    """
    if od_source not in data:
        logger.warning(f"OD source '{od_source}' not found in data")
        return [], []

    models = sorted(data[od_source].keys())
    vanilla_models = sorted([m for m in models if "vanilla" in m.lower()])
    distilled_models = sorted([m for m in models if "distill" in m.lower()])

    logger.info(
        f"Detected {len(vanilla_models)} vanilla and {len(distilled_models)} distilled models"
    )

    return vanilla_models, distilled_models


def generate_model_colors(models: List[str]) -> Dict[str, str]:
    """Generate distinct colors for models dynamically using consistent palette

    Args:
        models: List of model names

    Returns:
        Dictionary mapping model names to hex color codes
    """
    if not models:
        return {}

    # Use consistent seaborn palette for all model counts
    palette = sns.color_palette(MODEL_COLOR_PALETTE, len(models))
    hex_colors = [
        f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        for r, g, b in palette
    ]

    return {model: color for model, color in zip(models, hex_colors)}


def generate_model_labels(models: List[str]) -> Dict[str, str]:
    """Generate human-readable labels from model names

    Args:
        models: List of model names

    Returns:
        Dictionary mapping model names to display labels
    """
    if not models:
        return {}

    labels = {}
    for model in models:
        # Convert underscores to spaces and title case
        label = model.replace("_", " ").title()

        # Preserve seed numbers in parentheses
        if "seed" in model.lower():
            import re

            match = re.search(r"seed(\d+)", model, re.IGNORECASE)
            if match:
                seed_num = match.group(1)
                base_name = re.sub(r"_?seed\d+", "", model, flags=re.IGNORECASE)
                base_label = base_name.replace("_", " ").title()
                label = f"{base_label} (seed {seed_num})"

        labels[model] = label

    return labels


def get_model_colors(data: Dict, od_source: str = "train") -> Dict[str, str]:
    """Get color mapping for all models in dataset

    Convenience wrapper that combines classify_models() and generate_model_colors()
    for easy use in plotting functions.

    Args:
        data: Loaded scenario data
        od_source: OD source to analyze (default: 'train')

    Returns:
        Dictionary mapping model names to hex color codes
    """
    vanilla_models, distilled_models = classify_models(data, od_source)
    all_models = sorted(vanilla_models + distilled_models)
    return generate_model_colors(all_models)


def get_model_labels(data: Dict, od_source: str = "train") -> Dict[str, str]:
    """Get label mapping for all models in dataset

    Convenience wrapper that combines classify_models() and generate_model_labels()
    for easy use in plotting functions.

    Args:
        data: Loaded scenario data
        od_source: OD source to analyze (default: 'train')

    Returns:
        Dictionary mapping model names to display labels
    """
    vanilla_models, distilled_models = classify_models(data, od_source)
    all_models = sorted(vanilla_models + distilled_models)
    return generate_model_labels(all_models)


def get_available_scenarios(data: Dict, od_source: str = "train") -> List[str]:
    """Extract list of scenarios that actually exist in the data

    Args:
        data: Loaded scenario data
        od_source: OD source to analyze (default: 'train')

    Returns:
        Sorted list of scenario names that exist across models
    """
    if od_source not in data:
        logger.warning(f"OD source '{od_source}' not found in data")
        return []

    # Collect all unique scenario names across all models
    all_scenarios = set()

    for model, model_data in data[od_source].items():
        if "individual_scenarios" in model_data:
            all_scenarios.update(model_data["individual_scenarios"].keys())

    scenarios = sorted(list(all_scenarios))

    if scenarios:
        logger.info(
            f"Found {len(scenarios)} scenarios in data: {', '.join(scenarios[:5])}"
            + (f" ... and {len(scenarios) - 5} more" if len(scenarios) > 5 else "")
        )
    else:
        logger.warning("No scenarios found in data")

    return scenarios


def get_available_metrics(data: Dict, od_source: str = "train") -> List[str]:
    """Extract list of metrics that actually exist in the data

    Args:
        data: Loaded scenario data
        od_source: OD source to analyze (default: 'train')

    Returns:
        Sorted list of metric names that exist across scenarios
    """
    if od_source not in data:
        logger.warning(f"OD source '{od_source}' not found in data")
        return []

    # Collect all unique metric names across all models and scenarios
    all_metrics = set()

    for model, model_data in data[od_source].items():
        if "individual_scenarios" in model_data:
            for scenario, scenario_data in model_data["individual_scenarios"].items():
                if "metrics" in scenario_data:
                    all_metrics.update(scenario_data["metrics"].keys())

    metrics = sorted(list(all_metrics))

    if metrics:
        logger.info(f"Found {len(metrics)} metrics in data: {', '.join(metrics)}")
    else:
        logger.warning("No metrics found in data")

    return metrics


def get_metric_display_labels(metrics: List[str]) -> List[str]:
    """Generate human-readable display labels for metrics dynamically

    Uses intelligent heuristics to format any metric name into a compact display label.
    Handles common patterns like suffixes (JSD, km, mean) and splits long names.

    Args:
        metrics: List of metric names

    Returns:
        List of formatted labels for display (same order as input)
    """
    if not metrics:
        return []

    # Common unit patterns to wrap in parentheses
    unit_suffixes = {
        "_km": "(km)",
        "_m": "(m)",
        "_ms": "(ms)",
        "_s": "(s)",
        "_pct": "(%)",
        "_deg": "(°)",
    }

    labels = []
    for metric in metrics:
        # Check for unit suffixes first
        formatted = False
        for suffix, unit in unit_suffixes.items():
            if metric.endswith(suffix):
                base = metric[: -len(suffix)]
                # Convert base to readable form
                readable_base = base.replace("_", " ").title()
                label = f"{readable_base}\n{unit}"
                labels.append(label)
                formatted = True
                break

        if formatted:
            continue

        # Handle underscores generically - split into max 2 parts for compactness
        if "_" in metric:
            parts = metric.split("_")

            # Special case: Keep common suffixes as-is (JSD, MAE, etc.)
            if len(parts) >= 2 and parts[-1].isupper() and len(parts[-1]) <= 4:
                # Distance_JSD -> Distance\nJSD
                base = "_".join(parts[:-1])
                suffix = parts[-1]
                readable_base = base.replace("_", " ").title()
                label = f"{readable_base}\n{suffix}"
            elif len(parts) == 2:
                # Simple two-part: Distance_mean -> Distance\nMean
                label = f"{parts[0].title()}\n{parts[1].title()}"
            else:
                # Complex multi-part: Split in middle for balance
                mid = len(parts) // 2
                first_half = " ".join(parts[:mid]).title()
                second_half = " ".join(parts[mid:]).title()
                label = f"{first_half}\n{second_half}"

            labels.append(label)
        else:
            # No underscore - use as-is
            labels.append(metric)

    return labels
