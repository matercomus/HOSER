# HOSER Distillation Documentation

Comprehensive documentation for the HOSER knowledge distillation project.

## Table of Contents

- [Quick Start](#quick-start)
- [Core Documentation](#core-documentation)
- [Guides](#guides)
- [Reference](#reference)
- [Evaluation](#evaluation)
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

---

## Documentation Maintenance

### Adding New Documentation

When adding new documentation:

1. Place core implementation docs in `docs/`
2. Place how-to guides in `docs/guides/`
3. Place technical references in `docs/reference/`
4. Place evaluation-specific docs in `hoser-distill-optuna-6/`
5. Move outdated docs to `docs/archive/`
6. Update this README with links and descriptions

### File Organization Principles

- **Core docs** (`docs/`): Essential reading for understanding the project
- **Guides** (`docs/guides/`): Practical how-to content
- **Reference** (`docs/reference/`): Technical deep-dives and lookup material
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
│   │   └── WANDB_CLI_EXAMPLES.md  # WandB CLI usage
│   ├── reference/
│   │   ├── PREPROCESSING_ANALYSIS.md
│   │   └── MODEL_LOCATIONS.md
│   └── archive/
│       ├── LMTAD-Critic.md
│       ├── DEBUGGING_NOTES.md
│       └── PRIVACY_ANALYSIS.md
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

**Last Updated:** October 14, 2025  
**Documentation Version:** 1.0 (post-reorganization)

