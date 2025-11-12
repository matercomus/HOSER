# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HOSER (Hierarchical Origin-destination Spatio-temporal Encoder-decoder for Roads) is a research project focused on knowledge distillation from LM-TAD (teacher model) into a more efficient student model for trajectory generation.

## Development Setup

### Environment Setup
```bash
# Requires Python 3.12+
uv venv  # Create virtual environment
uv pip install -e ".[dev]"  # Install with development dependencies
pre-commit install  # Set up code quality hooks
```

### Package Management
This project uses `uv` exclusively for dependency management. Never use pip, pip-tools, or poetry directly.

```bash
# Add or upgrade dependencies
uv add <package>

# Remove dependencies
uv remove <package>

# Reinstall all dependencies from lock file
uv sync
```

### Common Commands

#### Testing
```bash
pytest tests/  # Run all tests
pytest tests/test_specific.py  # Run specific test file
pytest tests/test_specific.py::test_function  # Run specific test function
```

#### Code Quality
```bash
pre-commit run --all-files  # Run all pre-commit hooks

# Linting with Ruff
uv tool run ruff check .  # Check for issues
uv tool run ruff check --fix .  # Auto-fix issues

# Formatting with Ruff
uv tool run ruff format .  # Format code
```

Run code quality checks before:
- Committing code changes
- Creating pull requests
- Marking tasks as complete
- Running long-running scripts

## Architecture Overview

### Core Components

1. HOSER Model (`hoser.py`)
   - RoadNetworkEncoder: Processes road network topology and features
   - TrajectoryEncoder: Handles trajectory sequences
   - Navigator: Core navigation logic for predictions

2. Training System (`train.py`)
   - Mixed precision training with gradient clipping
   - Learning rate warmup and cosine decay
   - TensorBoard monitoring integration
   - Loss components: next step prediction and travel time estimation

3. Data Pipeline (`dataset.py`)
   - Processes road network geometry and topology
   - Handles trajectory data with smart caching
   - Computes spatial and temporal features
   - Memory-aware data loading with disk fallback

### Key Workflows

1. Model Training Pipeline:
   - Data preprocessing and feature computation
   - Model training with validation-based selection
   - Performance monitoring and checkpointing
   - Result logging and model persistence

2. Evaluation System:
   - Next step prediction accuracy metrics
   - Travel time prediction MAPE
   - Three-way dataset split (train/val/test)
   - Comprehensive logging and analysis

3. Data Processing:
   - Road network graph construction
   - Trajectory to road sequence conversion
   - Spatio-temporal feature computation
   - Memory-efficient data handling

## Project Configuration

Dependencies are managed through pyproject.toml with key requirements:
- PyTorch ecosystem: torch, torch-geometric
- Data processing: pandas, polars, numpy
- Geospatial: geopandas, geopy, shapely
- Visualization: matplotlib, plotly, seaborn
- Monitoring: wandb, tensorboard

## Long-Running Jobs

The project involves several types of long-running tasks (>5 minutes):
- Model training
- Hyperparameter optimization (Optuna)
- Large dataset processing
- Batch inference

For these tasks:
1. Always run in a tmux/screen session
2. Monitor progress via:
   - WandB dashboard
   - Optuna database queries
   - Log files
3. Use provided tmux commands:
   ```bash
   tmux new-session -s hoser-job  # Start new session
   # Run your long command
   # Detach: Ctrl+b, then d
   tmux attach-session -t hoser-job  # Reattach
   tmux ls  # List sessions
   ```

## Documentation Structure

Key documentation in `docs/`:
- Research methodology (LMTAD-Distillation.md)
- Architecture specifications
- Model checkpointing strategy
- Evaluation and analysis guides