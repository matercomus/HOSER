"""
Data loading utilities for scenario analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import seaborn as sns

# Consistent color palette for all model visualizations
MODEL_COLOR_PALETTE = "husl"


class ScenarioDataLoader:
    """Load and manage scenario analysis data"""
    
    def __init__(self, eval_dir: Path):
        self.eval_dir = Path(eval_dir).resolve()
        self.scenarios_dir = self.eval_dir / 'scenarios'
        
        if not self.scenarios_dir.exists():
            raise FileNotFoundError(f"Scenarios directory not found: {self.scenarios_dir}")
    
    def load_all(self) -> Dict:
        """Load all scenario data from JSON files"""
        logger.info(f"📂 Loading scenario data from {self.eval_dir}")
        
        data = {}
        
        # Auto-detect OD sources
        for od_dir in self.scenarios_dir.iterdir():
            if od_dir.is_dir() and not od_dir.name.startswith('.'):
                od_source = od_dir.name
                data[od_source] = {}
                
                # Load each model
                for model_dir in od_dir.iterdir():
                    if model_dir.is_dir():
                        model = model_dir.name
                        metrics_file = model_dir / 'scenario_metrics.json'
                        
                        if metrics_file.exists():
                            with open(metrics_file) as f:
                                data[od_source][model] = json.load(f)
                            logger.info(f"  ✅ {od_source}/{model}")
        
        if not data:
            raise ValueError("No scenario data found!")
        
        return data
    
    def get_common_scenarios(self, data: Dict, od_source: str = 'train') -> List[str]:
        """Get scenarios that exist across all models"""
        if od_source not in data:
            return []
        
        # Get scenarios from first model
        first_model = list(data[od_source].keys())[0]
        scenarios = set(data[od_source][first_model].get('individual_scenarios', {}).keys())
        
        # Intersect with other models
        for model in data[od_source].keys():
            model_scenarios = set(data[od_source][model].get('individual_scenarios', {}).keys())
            scenarios &= model_scenarios
        
        # Return in consistent order
        return sorted(list(scenarios))
    
    def get_models(self, data: Dict, od_source: str = 'train') -> List[str]:
        """Get list of models for a given OD source"""
        if od_source not in data:
            return []
        return sorted(data[od_source].keys())


def get_scenario_list(data: Dict, od_source: str = 'train', model: str = 'vanilla') -> List[str]:
    """Extract ordered list of scenarios"""
    if od_source not in data or model not in data[od_source]:
        return []
    
    scenarios = data[od_source][model].get('individual_scenarios', {}).keys()
    return sorted(list(scenarios))


def get_metric_value(data: Dict, od_source: str, model: str, 
                     scenario: str, metric: str) -> Optional[float]:
    """Safe metric extraction"""
    try:
        return data[od_source][model]['individual_scenarios'][scenario]['metrics'][metric]
    except KeyError:
        logger.warning(f"Metric not found: {od_source}/{model}/{scenario}/{metric}")
        return None


def calculate_improvement(data: Dict, od_source: str, scenario: str, metric: str,
                         baseline: str = 'vanilla', 
                         improved: str = 'distilled_seed44') -> Optional[float]:
    """Calculate percentage improvement (lower is better for all metrics)"""
    baseline_val = get_metric_value(data, od_source, baseline, scenario, metric)
    improved_val = get_metric_value(data, od_source, improved, scenario, metric)
    
    if baseline_val is None or improved_val is None or baseline_val == 0:
        return None
    
    # For metrics where lower is better
    improvement = ((baseline_val - improved_val) / baseline_val) * 100
    return improvement


def classify_models(data: Dict, od_source: str = 'train') -> Tuple[List[str], List[str]]:
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
    vanilla_models = sorted([m for m in models if 'vanilla' in m.lower()])
    distilled_models = sorted([m for m in models if 'distill' in m.lower()])
    
    logger.info(f"Detected {len(vanilla_models)} vanilla and {len(distilled_models)} distilled models")
    
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
    hex_colors = [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' 
                 for r, g, b in palette]
    
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
        label = model.replace('_', ' ').title()
        
        # Preserve seed numbers in parentheses
        if 'seed' in model.lower():
            import re
            match = re.search(r'seed(\d+)', model, re.IGNORECASE)
            if match:
                seed_num = match.group(1)
                base_name = re.sub(r'_?seed\d+', '', model, flags=re.IGNORECASE)
                base_label = base_name.replace('_', ' ').title()
                label = f"{base_label} (seed {seed_num})"
        
        labels[model] = label
    
    return labels


def get_available_scenarios(data: Dict, od_source: str = 'train') -> List[str]:
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
        if 'individual_scenarios' in model_data:
            all_scenarios.update(model_data['individual_scenarios'].keys())
    
    scenarios = sorted(list(all_scenarios))
    
    if scenarios:
        logger.info(f"Found {len(scenarios)} scenarios in data: {', '.join(scenarios[:5])}"
                   + (f" ... and {len(scenarios)-5} more" if len(scenarios) > 5 else ""))
    else:
        logger.warning("No scenarios found in data")
    
    return scenarios


def get_available_metrics(data: Dict, od_source: str = 'train') -> List[str]:
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
        if 'individual_scenarios' in model_data:
            for scenario, scenario_data in model_data['individual_scenarios'].items():
                if 'metrics' in scenario_data:
                    all_metrics.update(scenario_data['metrics'].keys())
    
    metrics = sorted(list(all_metrics))
    
    if metrics:
        logger.info(f"Found {len(metrics)} metrics in data: {', '.join(metrics)}")
    else:
        logger.warning("No metrics found in data")
    
    return metrics


def get_metric_display_labels(metrics: List[str]) -> List[str]:
    """Generate human-readable display labels for metrics
    
    Args:
        metrics: List of metric names
        
    Returns:
        List of formatted labels for display (same order as input)
    """
    if not metrics:
        return []
    
    labels = []
    for metric in metrics:
        # Common metric formatting rules
        if metric.endswith('_JSD'):
            # Distance_JSD -> Distance\nJSD
            base = metric[:-4]
            label = f"{base}\nJSD"
        elif metric.endswith('_km'):
            # Hausdorff_km -> Hausdorff\n(km)
            base = metric[:-3]
            label = f"{base}\n(km)"
        elif metric.endswith('_mean'):
            # Distance_mean -> Distance\nMean
            base = metric[:-5]
            label = f"{base}\nMean"
        elif '_' in metric:
            # Generic: split on underscore
            parts = metric.split('_')
            if len(parts) == 2:
                label = f"{parts[0]}\n{parts[1]}"
            else:
                label = metric.replace('_', '\n')
        else:
            # No special formatting
            label = metric
        
        labels.append(label)
    
    return labels

