# HOSER Distillation Documentation

Comprehensive documentation for the HOSER knowledge distillation project.

## Table of Contents

- [Quick Start](#quick-start)
- [Core Documentation](#core-documentation)
- [Guides](#guides)
- [Reference](#reference)
- [Results](#results)
- [Evaluation](#evaluation)
- [Logs](#logs)
- [Archive](#archive)

---

## Quick Start

New to the project? Start here:

1. **[LMTAD-Distillation.md](LMTAD-Distillation.md)** - Main comprehensive guide (2610 lines)
   - What is knowledge distillation and why it matters
   - Complete technical explanation with worked examples
   - Implementation details and loss formulation
   - Hyperparameter tuning with Optuna
   - Evaluation methodology and trajectory generation

2. **[DATASET_SETUP.md](DATASET_SETUP.md)** - Dataset preparation guide
   - Step-by-step setup for new datasets (Beijing, Porto, etc.)
   - Prerequisites and required files
   - Preprocessing scripts usage
   - Configuration file creation
   - Troubleshooting common issues

---

## Core Documentation

### Main Distillation Guide

**[LMTAD-Distillation.md](LMTAD-Distillation.md)**

The comprehensive 2610-line guide covering:

- **Executive Summary** - Quick overview of the approach
- **What's Implemented** - All current components
- **Loss Design** - Mathematical formulation with worked examples
- **Detailed Worked Example** - Step-by-step through a real training iteration
- **Hyperparameter Tuning** - Optuna 2-phase optimization strategy
- **Evaluation** - Trajectory generation and metrics
- **Implementation Details** - All code components explained

**Recommended reading order:**
1. Executive Summary (understand the approach)
2. Quick Start (get training immediately)
3. Loss Design (understand the math)
4. Detailed Worked Example (see it in action)
5. Implementation Details (when debugging or extending)

### Dataset Setup

**[DATASET_SETUP.md](DATASET_SETUP.md)**

Complete guide for setting up new datasets:

- Dataset structure requirements
- Road network partitioning (KaHIP)
- Zone transition matrix generation
- LM-TAD teacher weights extraction
- Configuration file creation
- Memory optimization for different trajectory lengths
- Full Porto dataset example

---

## Guides

Practical how-to guides for common tasks.

### WandB CLI Usage

**[guides/WANDB_CLI_EXAMPLES.md](guides/WANDB_CLI_EXAMPLES.md)**

Quick reference for downloading models and managing WandB runs:

- Download models using WandB CLI
- Search for runs programmatically
- Compare vanilla vs distilled checkpoints
- WandB CLI vs Python API trade-offs

### Wang Statistical Abnormality Analysis

**[guides/RUN_WANG_ABNORMALITY_ANALYSIS.md](guides/RUN_WANG_ABNORMALITY_ANALYSIS.md)**

Step-by-step guide for running statistical abnormality detection based on Wang et al. 2018 methodology:

- Baseline computation for OD pairs
- Statistical abnormality detection configuration
- Cross-dataset evaluation workflow
- Quality filtering and interpretation

### Abnormal OD Pair Workflow

**[guides/ABNORMAL_OD_WORKFLOW.md](guides/ABNORMAL_OD_WORKFLOW.md)**

Complete pipeline for testing models on challenging abnormal trajectory scenarios:

- Extract abnormal OD pairs from real data
- Generate trajectories for challenging scenarios
- Evaluate model performance on edge cases
- Cross-dataset testing methodology

---

## Reference

Technical deep-dives and reference material.

### Preprocessing Analysis

**[reference/PREPROCESSING_ANALYSIS.md](reference/PREPROCESSING_ANALYSIS.md)**

Detailed analysis of HOSER's preprocessing pipeline (641 lines):

- Dataset structure and formats
- Road geometry file analysis
- Road relations and connectivity
- Zone partitioning methodology
- Trajectory data format
- Data encoding schemes
- POI integration strategies

### Model Locations

**[reference/MODEL_LOCATIONS.md](reference/MODEL_LOCATIONS.md)**

Reference for finding and using trained model checkpoints:

- Available models (vanilla baseline, distilled models)
- Model metadata (epochs, seeds, validation accuracy)
- Download instructions
- Generation and evaluation commands
- Model comparison summary

### Baseline Statistics

**[reference/BASELINE_STATISTICS.md](reference/BASELINE_STATISTICS.md)**

Methodology and results for computing OD-pair baseline statistics from real trajectory data:

- Baseline computation methodology
- Coverage statistics per dataset
- Distribution analysis
- Quality metrics and validation

### Road Network Mapping

**[reference/ROAD_NETWORK_MAPPING.md](reference/ROAD_NETWORK_MAPPING.md)**

Complete methodology for mapping road network IDs between datasets to enable cross-dataset analysis:

- Coordinate extraction and distance calculation
- Nearest neighbor matching algorithm
- Quality validation procedures
- Translation quality impact on results
- Cross-dataset analysis interpretation framework
- Known issues and solutions

---

## Evaluation

Complete evaluation methodology and results for the distillation experiments.

### Evaluation Subproject

**[../hoser-distill-optuna-6/](../hoser-distill-optuna-6/)**

The `hoser-distill-optuna-6/` directory contains all evaluation-related documentation and results:

- **[README.md](../hoser-distill-optuna-6/README.md)** - Evaluation pipeline overview and quick start
- **[EVALUATION_ANALYSIS.md](../hoser-distill-optuna-6/EVALUATION_ANALYSIS.md)** - Comprehensive results analysis (727 lines)
  - Experimental setup and fair comparison methodology
  - Complete results tables with all metrics
  - Key findings: path completion, trip length realism, spatial distribution quality
  - Why vanilla fails and what distillation transferred
  - Statistical analysis and conclusions
- **[PIPELINE_USAGE.md](../hoser-distill-optuna-6/PIPELINE_USAGE.md)** - Python evaluation pipeline guide
- **[OD_MATCHING_EXPLAINED.md](../hoser-distill-optuna-6/OD_MATCHING_EXPLAINED.md)** - OD pair matching methodology
- **[TRAJECTORY_VISUALIZATION_GUIDE.md](../hoser-distill-optuna-6/TRAJECTORY_VISUALIZATION_GUIDE.md)** - Visualization tools
- **[VANILLA_FAILURE_ANALYSIS.md](../hoser-distill-optuna-6/VANILLA_FAILURE_ANALYSIS.md)** - Deep-dive on vanilla model failures
- **[RESULTS_ANALYSIS.md](../hoser-distill-optuna-6/RESULTS_ANALYSIS.md)** - Additional results analysis
- **[figures/README.md](../hoser-distill-optuna-6/figures/README.md)** - Generated figures and plots

**Key Evaluation Findings:**
- Distilled models achieve 85-89% path completion success vs vanilla's 12-18%
- 87% reduction in distance distribution error (JSD)
- 98% reduction in spatial complexity error (radius of gyration JSD)
- Distilled models generate realistic 6.4 km trips vs vanilla's unrealistic 2.4 km trips

---

## Results

Main research results and analysis reports.

### Wang Statistical Abnormality Detection Results

**[results/WANG_ABNORMALITY_DETECTION_RESULTS.md](results/WANG_ABNORMALITY_DETECTION_RESULTS.md)**

Comprehensive analysis of statistical abnormality detection using Wang et al. 2018 methodology:

- Executive summary and methodology
- Experimental setup across Beijing and Porto datasets
- Real vs generated abnormality rate comparisons
- Model realism rankings
- Pattern distribution analysis (Abp1-4)
- Statistical significance tests
- Cross-dataset transfer performance
- Visualizations and detailed findings

---

## Logs

All execution logs are organized in `docs/logs/` for easy access and debugging.

### Log Organization

- **`logs/evaluation/beijing/`**: Evaluation pipeline logs for Beijing dataset
- **`logs/evaluation/porto/`**: Evaluation pipeline logs for Porto dataset
- **`logs/baselines/`**: Baseline computation logs
- **`logs/pipeline/`**: Root-level pipeline execution logs

Logs are timestamped and organized by dataset and operation type for easy navigation.

---

## Archive

Historical and outdated documentation kept for reference.

### Outdated Approaches

**[archive/LMTAD-Critic.md](archive/LMTAD-Critic.md)** (Sep 26, 2024)

Early exploration of using LM-TAD as a real-time "co-pilot" or critic during trajectory generation. This approach was superseded by the knowledge distillation approach (training-time transfer instead of inference-time guidance).

**Historical interest:** Shows the evolution from inference-time integration to training-time distillation.

### Resolved Issues

**[archive/DEBUGGING_NOTES.md](archive/DEBUGGING_NOTES.md)** (Sep 24, 2024)

Dataset loading errors and connectivity analysis from early development. Issues were resolved by identifying disconnected roads in the Beijing dataset.

**Historical interest:** Documents the dataset quality challenges and debugging process.

### Privacy Analysis

**[archive/PRIVACY_ANALYSIS.md](archive/PRIVACY_ANALYSIS.md)** (Sep 24, 2024)

Analysis of origin-destination (OD) pair preservation in trajectory generation and privacy implications. Notes that HOSER (like many mobility generation models) uses real OD pairs from training data, which is standard practice but has privacy considerations.

**Historical interest:** Important context on the trade-offs between realism and privacy in trajectory generation research.

### Completed Implementation Plans

**[archive/WANG_STATISTICAL_DETECTION_PLAN.md](archive/WANG_STATISTICAL_DETECTION_PLAN.md)** (Nov 2025)

Implementation plan for statistical abnormality detection based on Wang et al. 2018. Documents the phased implementation approach, methodology decisions, and all completed tasks.

**Historical interest:** Shows the evolution from threshold-based to statistical abnormality detection.

**[archive/WANG_IMPLEMENTATION_SUMMARY.md](archive/WANG_IMPLEMENTATION_SUMMARY.md)** (Nov 2025)

Concise summary of the completed Wang statistical detection implementation, including key features and changes.

### Superseded Analysis

**[archive/Z_SCORE_RESULTS_ANALYSIS.md](archive/Z_SCORE_RESULTS_ANALYSIS.md)** (Nov 2025)

Analysis of the previous z-score based abnormality detection results, which were found to be inaccurate (all 0% due to expecting GPS coordinates but receiving road IDs). Superseded by the Wang statistical method.

**Historical interest:** Documents why the z-score method was replaced and justifies the decision to use only the Wang statistical method.

**[archive/ABNORMAL_CROSS_DATASET.md](archive/ABNORMAL_CROSS_DATASET.md)** (Nov 2025)

Original notes on cross-dataset analysis and false positive discovery. Content has been merged into `reference/ROAD_NETWORK_MAPPING.md` for better organization.

**Historical interest:** Documents the discovery of the 99% false positive issue and the solution approach.

---

## Documentation Maintenance

### Adding New Documentation

When adding new documentation:

1. Place core implementation docs in `docs/`
2. Place how-to guides in `docs/guides/`
3. Place technical references in `docs/reference/`
4. Place research results in `docs/results/`
5. Place execution logs in `docs/logs/` (organized by type and dataset)
6. Place evaluation-specific docs in `hoser-distill-optuna-6/`
7. Move outdated docs to `docs/archive/`
8. Update this README with links and descriptions

### File Organization Principles

- **Core docs** (`docs/`): Essential reading for understanding the project
- **Guides** (`docs/guides/`): Practical how-to content
- **Reference** (`docs/reference/`): Technical deep-dives and lookup material
- **Results** (`docs/results/`): Final research results and analysis reports
- **Logs** (`docs/logs/`): Execution logs organized by dataset and operation type
- **Evaluation** (`hoser-distill-optuna-6/`): Results, analysis, and evaluation methodology
- **Archive** (`docs/archive/`): Historical content, outdated approaches, resolved issues

---

## Project Structure

```
HOSER/
├── docs/                           # Main documentation directory
│   ├── README.md                   # This file - navigation hub
│   ├── LMTAD-Distillation.md      # Main comprehensive guide
│   ├── DATASET_SETUP.md           # Dataset preparation guide
│   ├── guides/
│   │   ├── WANDB_CLI_EXAMPLES.md
│   │   ├── RUN_WANG_ABNORMALITY_ANALYSIS.md
│   │   └── ABNORMAL_OD_WORKFLOW.md
│   ├── reference/
│   │   ├── PREPROCESSING_ANALYSIS.md
│   │   ├── MODEL_LOCATIONS.md
│   │   ├── BASELINE_STATISTICS.md
│   │   └── ROAD_NETWORK_MAPPING.md
│   ├── results/
│   │   └── WANG_ABNORMALITY_DETECTION_RESULTS.md
│   ├── logs/
│   │   ├── evaluation/
│   │   │   ├── beijing/
│   │   │   └── porto/
│   │   ├── baselines/
│   │   └── pipeline/
│   └── archive/
│       ├── LMTAD-Critic.md
│       ├── DEBUGGING_NOTES.md
│       ├── PRIVACY_ANALYSIS.md
│       ├── WANG_STATISTICAL_DETECTION_PLAN.md
│       ├── WANG_IMPLEMENTATION_SUMMARY.md
│       ├── Z_SCORE_RESULTS_ANALYSIS.md
│       └── ABNORMAL_CROSS_DATASET.md
├── hoser-distill-optuna-6/         # Evaluation subproject
│   ├── README.md
│   ├── EVALUATION_ANALYSIS.md      # Main results analysis
│   ├── PIPELINE_USAGE.md
│   ├── OD_MATCHING_EXPLAINED.md
│   ├── TRAJECTORY_VISUALIZATION_GUIDE.md
│   ├── VANILLA_FAILURE_ANALYSIS.md
│   ├── RESULTS_ANALYSIS.md
│   └── figures/
│       └── README.md
├── README.md                       # Project root README
├── CHANGELOG.md                    # Version history
└── ... (source code)
```

---

**Last Updated:** November 4, 2025  
**Documentation Version:** 2.0 (post-cleanup and reorganization)

