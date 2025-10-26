"""
Data loading utilities for scenario analysis.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


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

