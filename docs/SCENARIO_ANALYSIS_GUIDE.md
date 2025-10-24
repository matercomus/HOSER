# Scenario-Based Trajectory Analysis Guide

## Overview

The scenario analysis feature provides post-processing analysis of generated trajectories based on temporal, spatial, and functional scenarios. This allows for deeper insights into model performance across different trip types and conditions.

## Architecture

The feature is designed as a **standalone post-processing pipeline** that:
- Runs after normal generation and evaluation
- Has zero impact on inference performance  
- Is highly configurable via YAML files
- Provides hierarchical breakdowns and statistical analysis

## Configuration

### Master Templates

Scenario configurations are stored in the `config/` directory:
- `config/scenarios_beijing.yaml` - Beijing-specific scenarios
- `config/scenarios_porto.yaml` - Porto-specific scenarios

### Configuration Structure

```yaml
# Temporal scenarios
temporal:
  peak_hours:
    enabled: true
    weekday_morning: ["07:00", "09:00"]
    weekday_evening: ["17:00", "19:00"]
    
# Spatial scenarios  
spatial:
  city_center:
    enabled: true
    center_lat: 39.9042  # Beijing example
    center_lon: 116.4074
    radius_km: 5.0
    
  airport:
    enabled: true
    airports:
      - name: "Beijing Capital International Airport"
        lat: 40.0799
        lon: 116.6031
        radius_km: 2.0
        
# Analysis settings
analysis:
  min_samples_per_scenario: 10
  reporting:
    individual_scenarios: true
    combinations: true
    hierarchical: true
  statistics:
    enabled: true
```

## Usage

### Integrated with Pipeline (Recommended)

```bash
# Run full pipeline with scenario analysis
cd hoser-distill-optuna-6-evaluation-abc123
uv run python ../hoser-distill-optuna-6/python_pipeline.py --run-scenarios

# Use custom scenarios config
uv run python ../hoser-distill-optuna-6/python_pipeline.py \
    --run-scenarios \
    --scenarios-config config/scenarios_beijing_custom.yaml
```

### Standalone Analysis

```bash
# Analyze existing evaluation directory
uv run python tools/analyze_scenarios.py \
    --eval-dir hoser-distill-optuna-6-evaluation-abc123 \
    --config config/scenarios_beijing.yaml

# Analyze specific models only
uv run python tools/analyze_scenarios.py \
    --eval-dir evaluation_xyz \
    --config config/scenarios_porto.yaml \
    --models vanilla,distilled
```

## Output Structure

After running scenario analysis, the evaluation directory will contain:

```
evaluation_directory/
├── scenarios/
│   ├── test/
│   │   ├── vanilla/
│   │   │   ├── scenario_analysis.json     # Detailed results
│   │   │   ├── scenario_distribution.png  # Distribution plot
│   │   │   └── metric_comparison.png      # Heatmap of metrics
│   │   └── distilled/
│   │       └── ...
│   └── train/
│       └── ...
```

## Results Format

The `scenario_analysis.json` contains:

```json
{
  "overview": {
    "total_trajectories": 5000,
    "scenario_distribution": {
      "peak": 1247,
      "off_peak": 3753,
      "weekday": 3571,
      "weekend": 1429,
      "airport": 312,
      "city_center": 2145
    }
  },
  "individual_scenarios": {
    "peak": {
      "count": 1247,
      "percentage": 24.9,
      "metrics": {
        "Distance_JSD": 0.1876,
        "Duration_JSD": 0.2134,
        "Hausdorff_mean": 245.34
      }
    }
  },
  "hierarchical": {
    "airport": {
      "peak": {...},
      "off_peak": {...}
    }
  }
}
```

## Scenario Types

### Temporal Scenarios
- **Peak vs Off-Peak**: Defined by configurable time windows
- **Weekday vs Weekend**: Based on day of week

### Spatial Scenarios
- **City Center**: Trips to/from/within city center
- **Airport**: Trips involving airport locations  

### Functional Trip Types
- `to_airport`: Destination is airport
- `from_airport`: Origin is airport
- `to_center`: Trip ending in city center
- `from_center`: Trip starting from city center
- `within_center`: Both origin and destination in center
- `suburban`: All other trips

## Analysis Features

### Individual Scenario Metrics
Each scenario is analyzed independently, computing all standard evaluation metrics (JSD, Hausdorff, etc.) for trajectories matching that scenario.

### Scenario Combinations
The system analyzes combinations like "peak+weekday+airport" to understand compound effects.

### Hierarchical Breakdowns
For deeper insights, scenarios are broken down hierarchically:
- Airport trips → Peak vs Off-peak airport trips
- City center trips → Peak vs Off-peak center trips
- Weekday trips → By trip type (to/from airport, to/from center)

### Statistical Analysis
Basic statistical comparisons between scenarios identify significant performance differences.

## Customization

### Adding New Scenarios

1. Edit the appropriate config file
2. Add new categorization logic in `ScenarioCategorizer` 
3. Update hierarchical breakdowns if needed

### Modifying Metrics

The analysis reuses existing evaluation metrics from `evaluation.py`. To add new metrics, update the evaluation module.

## Performance Considerations

- Scenario analysis is memory-efficient, processing trajectories in batches
- Visualization generation can be disabled if not needed
- The `min_samples_per_scenario` setting prevents computing metrics for rare scenarios

## Integration with Setup Script

The `setup_evaluation.py` script automatically:
1. Copies the appropriate scenarios config template
2. Places it in the evaluation directory
3. Allows for per-evaluation customization

This ensures reproducibility and allows different scenario definitions for different experiments.
