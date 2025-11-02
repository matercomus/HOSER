<!-- c4eaffdb-b25a-4d0b-9f4b-1c1559690088 41f2d5e5-270e-445b-aff5-4e105c43986a -->
# Phase Decorator Pipeline Architecture

## Overview

Replace boolean skip flags with elegant phase-based architecture using decorators and registry pattern for cleaner, extensible pipeline control.

**Target**: 50% code reduction (200 → 100 lines of phase logic)

---

## Current Status Summary

**Last Updated:** Not started

**Completed Tasks:** 0/10

**Active Agents:** None

**Blocked Tasks:** None

---

## Phase 1: Foundation (Parallel Safe)

### Task 1.1: Add Phase Decorator Infrastructure [ ]

**Files:** `python_pipeline.py` (lines 73-92, after imports)

**Dependencies:** None

**Worktree:** Any (e.g., `HOSER-phase1-foundation`)

**Branch:** `feat/phase-decorator-foundation`

**Description:**

Add decorator and registry for auto-registering pipeline phases.

**Implementation:**

```python
# Add after imports, before PipelineConfig class
from functools import wraps
from typing import Callable, Dict

# Phase registry - auto-populated by decorator
PHASE_REGISTRY: Dict[str, Callable] = {}

def phase(name: str, critical: bool = False):
    """Decorator to register a pipeline phase
    
    Args:
        name: Phase identifier (used in CLI)
        critical: If True, pipeline stops on failure
    """
    def decorator(func):
        PHASE_REGISTRY[name] = {
            'func': func,
            'critical': critical,
            'name': name
        }
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

**Validation:**

```bash
# Check imports
grep "from functools import wraps" python_pipeline.py

# Check registry defined
grep "PHASE_REGISTRY" python_pipeline.py

# Syntax check
uv tool run ruff check python_pipeline.py

# Quick import test
uv run python -c "import sys; sys.path.insert(0, '.'); from python_pipeline import PHASE_REGISTRY; print('OK')"
```

**Commit:** `feat: add phase decorator infrastructure for pipeline`

---

### Task 1.2: Add Phases Field to Config [ ]

**Files:** `python_pipeline.py` (PipelineConfig class, lines ~89-132)

**Dependencies:** None (parallel with 1.1)

**Worktree:** Any (e.g., `HOSER-phase1-config`)

**Branch:** `feat/phase-config-field`

**Description:**

Add phase registry field to config, keep old skip_* fields for backward compatibility.

**Implementation:**

```python
@dataclass
class PipelineConfig:
    # NEW: Phase-based control
    phases: Set[str] = field(default_factory=lambda: {
        "generation", "base_eval", "cross_dataset", "abnormal"
    })
    
    # DEPRECATED but kept for backward compat
    skip_gene: bool = False
    skip_eval: bool = False
    # ... other existing fields ...
```

**Validation:**

```bash
# Check field added
grep "phases: Set\[str\]" python_pipeline.py

# Syntax check
uv tool run ruff check python_pipeline.py

# Quick test
uv run python -c "from python_pipeline import PipelineConfig; c = PipelineConfig(); print(c.phases)"
```

**Commit:** `feat: add phases field to PipelineConfig`

---

## Phase 2: Phase Method Extraction (Parallel Safe)

All tasks in Phase 2 can run in parallel - each creates a new method in different sections.

### Task 2.1: Extract Generation Phase Method [ ]

**Files:** `python_pipeline.py` (new method in EvaluationPipeline, ~line 1350)

**Dependencies:** Task 1.1 (needs decorator)

**Worktree:** `HOSER-phase2-generation`

**Branch:** `feat/phase-method-generation`

**Description:**

Extract generation loop from run() into decorated phase method.

**Implementation:**

```python
# Add to EvaluationPipeline class
@phase("generation", critical=True)
def run_generation(self):
    """Generate trajectories for all models"""
    logger.info("🔄 Generating trajectories...")
    
    # Extract from existing run() method (lines ~1111-1207)
    for model_type in self.config.models:
        model_file = self.detector.find_model_file(model_type)
        if not model_file:
            continue
            
        for od_source in self.config.od_sources:
            if self.interrupted:
                raise KeyboardInterrupt()
            
            existing_file = self._check_existing_results(model_type, od_source)
            # ... rest of generation logic ...
```

**Validation:**

```bash
# Check decorator applied
grep "@phase(\"generation\"" python_pipeline.py

# Check method exists
grep "def run_generation" python_pipeline.py

# Syntax check
uv tool run ruff check python_pipeline.py
```

**Commit:** `refactor: extract generation into phase method`

---

### Task 2.2: Extract Base Evaluation Phase Method [ ]

**Files:** `python_pipeline.py` (new method in EvaluationPipeline, ~line 1400)

**Dependencies:** Task 1.1 (needs decorator)

**Worktree:** `HOSER-phase2-baseeval`

**Branch:** `feat/phase-method-base-eval`

**Description:**

Extract base evaluation loop from run() into decorated phase method.

**Implementation:**

```python
# Add to EvaluationPipeline class
@phase("base_eval", critical=True)
def run_base_eval(self):
    """Evaluate on base dataset (Beijing)"""
    logger.info("📊 Evaluating on base dataset...")
    
    # Extract from existing run() method (lines ~1244-1293)
    for model_type in self.config.models:
        for od_source in self.config.od_sources:
            if self.interrupted:
                raise KeyboardInterrupt()
            
            # Find generated file
            gene_dir = Path(f"./gene/{self.config.dataset}/seed{self.config.seed}")
            # ... rest of evaluation logic ...
```

**Validation:**

```bash
# Check decorator applied
grep "@phase(\"base_eval\"" python_pipeline.py

# Check method exists
grep "def run_base_eval" python_pipeline.py

# Syntax check
uv tool run ruff check python_pipeline.py
```

**Commit:** `refactor: extract base eval into phase method`

---

### Task 2.3: Wrap Cross-Dataset Phase Method [ ]

**Files:** `python_pipeline.py` (new method in EvaluationPipeline, ~line 1450)

**Dependencies:** Task 1.1 (needs decorator)

**Worktree:** `HOSER-phase2-crossdata`

**Branch:** `feat/phase-method-cross-dataset`

**Description:**

Create wrapper for existing _run_cross_dataset_evaluation() method.

**Implementation:**

```python
# Add to EvaluationPipeline class
@phase("cross_dataset", critical=False)
def run_cross_dataset(self):
    """Evaluate on cross-dataset (BJUT)"""
    if not self.config.cross_dataset_eval:
        logger.info("Cross-dataset not configured, skipping")
        return
    
    logger.info("🌐 Evaluating on cross-dataset...")
    self._run_cross_dataset_evaluation()
```

**Validation:**

```bash
# Check decorator applied
grep "@phase(\"cross_dataset\"" python_pipeline.py

# Check method exists
grep "def run_cross_dataset" python_pipeline.py

# Syntax check
uv tool run ruff check python_pipeline.py
```

**Commit:** `refactor: wrap cross-dataset eval as phase`

---

### Task 2.4: Wrap Abnormal Detection Phase Method [ ]

**Files:** `python_pipeline.py` (new method in EvaluationPipeline, ~line 1470)

**Dependencies:** Task 1.1 (needs decorator)

**Worktree:** `HOSER-phase2-abnormal`

**Branch:** `feat/phase-method-abnormal`

**Description:**

Create wrapper for existing _run_abnormal_detection_analysis() method.

**Implementation:**

```python
# Add to EvaluationPipeline class
@phase("abnormal", critical=False)
def run_abnormal(self):
    """Detect abnormal trajectories"""
    if not self.config.run_abnormal_detection:
        logger.info("Abnormal detection not configured, skipping")
        return
    
    logger.info("🔍 Running abnormal detection...")
    self._run_abnormal_detection_analysis()
```

**Validation:**

```bash
# Check decorator applied
grep "@phase(\"abnormal\"" python_pipeline.py

# Check method exists
grep "def run_abnormal" python_pipeline.py

# Syntax check
uv tool run ruff check python_pipeline.py
```

**Commit:** `refactor: wrap abnormal detection as phase`

---

### Task 2.5: Wrap Scenario Analysis Phase Method [ ]

**Files:** `python_pipeline.py` (new method in EvaluationPipeline, ~line 1490)

**Dependencies:** Task 1.1 (needs decorator)

**Worktree:** `HOSER-phase2-scenarios`

**Branch:** `feat/phase-method-scenarios`

**Description:**

Create wrapper for existing _run_scenario_analysis() method.

**Implementation:**

```python
# Add to EvaluationPipeline class
@phase("scenarios", critical=False)
def run_scenarios(self):
    """Run scenario analysis"""
    if not self.config.run_scenarios:
        logger.info("Scenarios not configured, skipping")
        return
    
    logger.info("🎯 Running scenario analysis...")
    self._run_scenario_analysis()
```

**Validation:**

```bash
# Check decorator applied
grep "@phase(\"scenarios\"" python_pipeline.py

# Check method exists
grep "def run_scenarios" python_pipeline.py

# Syntax check
uv tool run ruff check python_pipeline.py
```

**Commit:** `refactor: wrap scenario analysis as phase`

---

## Phase 3: Integration (Sequential)

### Task 3.1: Replace run() with Phase Executor [ ]

**Files:** `python_pipeline.py` (EvaluationPipeline.run() method, lines ~1081-1350)

**Dependencies:** Tasks 1.1, 2.1, 2.2, 2.3, 2.4, 2.5 (needs all phase methods)

**Worktree:** `HOSER-phase3-integration`

**Branch:** `feat/phase-based-run-method`

**Description:**

Replace monolithic run() with elegant phase executor loop.

**Implementation:**

```python
def run(self):
    """Execute all enabled phases"""
    logger.info("Starting HOSER Distillation Evaluation Pipeline")
    logger.info(f"Configuration: {self.config.__dict__}")
    logger.info(f"Enabled phases: {sorted(self.config.phases)}")
    
    # Define phase execution order
    phase_order = ["generation", "base_eval", "cross_dataset", "abnormal", "scenarios"]
    
    for phase_name in phase_order:
        if phase_name not in self.config.phases:
            logger.info(f"⏭️  Skipping phase: {phase_name}")
            continue
        
        if phase_name not in PHASE_REGISTRY:
            logger.warning(f"⚠️  Phase not registered: {phase_name}")
            continue
        
        phase_info = PHASE_REGISTRY[phase_name]
        logger.info(f"\n{'='*70}")
        logger.info(f"🚀 Running phase: {phase_name}")
        logger.info(f"{'='*70}")
        
        try:
            phase_info['func'](self)
            logger.info(f"✅ Phase {phase_name} completed")
        except Exception as e:
            logger.error(f"❌ Phase {phase_name} failed: {e}")
            if phase_info['critical']:
                logger.error("Critical phase failed, stopping pipeline")
                raise
            else:
                logger.warning("Non-critical phase failed, continuing")
    
    logger.info("\n✅ Pipeline completed successfully!")
    return True
```

**Validation:**

```bash
# Check old run() replaced
wc -l python_pipeline.py  # Should be significantly shorter

# Syntax check
uv tool run ruff check python_pipeline.py

# Integration test (dry run)
uv run python python_pipeline.py --help

# Test phase skipping
uv run python -c "
from python_pipeline import PipelineConfig
c = PipelineConfig()
c.phases = {'cross_dataset'}
print('Phases:', c.phases)
"
```

**Commit:** `refactor: replace run() with phase-based executor`

---

### Task 3.2: Add CLI Phase Control Flags [ ]

**Files:** `python_pipeline.py` (main() function, lines ~1368-1512)

**Dependencies:** Task 1.2 (needs phases config field)

**Worktree:** `HOSER-phase3-cli`

**Branch:** `feat/phase-cli-interface`

**Description:**

Add --only and --skip flags for intuitive phase control.

**Implementation:**

```python
def main():
    parser = argparse.ArgumentParser()
    
    # ... existing arguments ...
    
    # NEW: Phase control flags
    parser.add_argument(
        "--only",
        type=str,
        help="Run only these phases (comma-separated). "
             "Available: generation,base_eval,cross_dataset,abnormal,scenarios. "
             "Example: --only cross_dataset,abnormal"
    )
    parser.add_argument(
        "--skip",
        type=str,
        help="Skip these phases (comma-separated). "
             "Example: --skip generation,base_eval"
    )
    
    # Keep backward compatibility
    parser.add_argument("--skip-gene", action="store_true",
                       help="Skip generation (alias for --skip generation)")
    parser.add_argument("--skip-eval", action="store_true",
                       help="Skip all evaluation (alias for --skip base_eval,cross_dataset,abnormal)")
    
    args = parser.parse_args()
    config = PipelineConfig(str(config_path), eval_dir)
    
    # Apply phase control (BEFORE other overrides)
    if args.only:
        # Explicit selection - clear and set only specified
        config.phases = set(p.strip() for p in args.only.split(","))
        logger.info(f"Running only phases: {config.phases}")
    
    if args.skip:
        # Remove specified phases
        skip_phases = set(p.strip() for p in args.skip.split(","))
        config.phases -= skip_phases
        logger.info(f"Skipping phases: {skip_phases}")
    
    # Backward compatibility shortcuts
    if args.skip_gene:
        config.phases.discard("generation")
    if args.skip_eval:
        config.phases -= {"base_eval", "cross_dataset", "abnormal"}
    
    # ... rest of existing main() code ...
```

**Validation:**

```bash
# Test help shows new flags
uv run python python_pipeline.py --help | grep -A2 "only"
uv run python python_pipeline.py --help | grep -A2 "skip"

# Test --only parsing
uv run python -c "
import sys; sys.argv = ['', '--only', 'cross_dataset,abnormal']
# Would need to mock argparse for full test
print('OK - would parse to: {cross_dataset, abnormal}')
"

# Syntax check
uv tool run ruff check python_pipeline.py
```

**Commit:** `feat: add --only and --skip CLI flags for phases`

---

## Phase 4: Cleanup (After Everything Works)

### Task 4.1: Remove Deprecated Skip Flags [ ]

**Files:** `python_pipeline.py` (PipelineConfig class, lines ~89-132)

**Dependencies:** Tasks 3.1, 3.2 (full system working)

**Worktree:** `HOSER-phase4-cleanup`

**Branch:** `cleanup/remove-deprecated-skip-flags`

**Description:**

Remove deprecated skip_* fields now that phase system is working.

**Implementation:**

1. Remove from PipelineConfig:

   - `skip_gene: bool = False`
   - `skip_eval: bool = False`
   - `skip_base_eval: bool = False` (if added in branch attempts)

2. Update docstrings to reflect phase-based system

3. Update any YAML config examples in comments

**Validation:**

```bash
# Check fields removed
grep "skip_gene" python_pipeline.py  # Should only be in --skip-gene shortcut
grep "skip_eval" python_pipeline.py  # Should only be in --skip-eval shortcut

# Full pipeline test
cd hoser-distill-optuna-6
uv run python ../python_pipeline.py \
  --eval-dir . \
  --only cross_dataset,abnormal \
  --config config/evaluation.yaml

# Syntax check
uv tool run ruff check python_pipeline.py
```

**Commit:** `cleanup: remove deprecated skip_* config fields`

---

## Usage After Implementation

### Your Use Case (Skip Base, Run Cross + Abnormal)

```bash
uv run python ../python_pipeline.py \
  --eval-dir . \
  --config config/evaluation.yaml \
  --only cross_dataset,abnormal \
  2>&1 | tee bjut_abnormal.log
```

### Other Common Cases

```bash
# Only generation
python pipeline.py --only generation

# Skip generation, run everything else
python pipeline.py --skip generation

# Backward compatibility (still works)
python pipeline.py --skip-gene --skip-eval
```

---

## Agent Coordination Notes

**File Conflict Zones:**

- Tasks 1.1 and 1.2: Different sections, safe parallel
- Tasks 2.1-2.5: Different methods, safe parallel (all add new methods)
- Tasks 3.1 and 3.2: Different functions (run() vs main()), safe parallel
- Task 4.1: Must wait for 3.1 and 3.2 to complete

**Recommended Worktree Assignments:**

- Agent 1: Phase 1 (1.1, 1.2) → Phase 3 (3.1, 3.2) → Phase 4
- Agent 2: Phase 2 odd (2.1, 2.3, 2.5)
- Agent 3: Phase 2 even (2.2, 2.4)

**Merge Strategy:**

1. Merge Phase 1 tasks → test
2. Merge Phase 2 tasks → test
3. Merge Phase 3 tasks → test
4. Merge Phase 4 → final validation

### To-dos

- [ ] Add --skip-base-eval argument to argparse in main()
- [ ] Add skip_base_eval field to PipelineConfig class
- [ ] Add config override in main() for skip_base_eval flag
- [ ] Wrap base dataset evaluation loop with skip_base_eval conditional