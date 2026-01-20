# Phase 7: Analysis, Interpretability, and Failure Diagnostics

## Overview
Phase 7 focuses on analyzing the behavior of the agents trained in previous phases, with a focus on subgoal semantics, failure modes, gating diagnostics, and representation learning.

## Components

### 1. Data Collection
`src/analysis/collect_traces.py`
Runs agents (loaded from checkpoints or mock) in the environment and logs detailed episode traces to JSONL files.
Usage:
```bash
python src/analysis/collect_traces.py --agent <agent_type> --task <task_id> --id <run_id> --episodes <N> --noise_type <type> --noise_level <level>
```

### 2. Analysis Axes

*   **Axis A: Subgoal Semantics** (`src/analysis/axis_a_subgoals.py`)
    *   Analyzes the distribution of generated subgoals (Action, Object, Color).
    *   Computes reuse statistics and entropy.
*   **Axis B: Failure Modes** (`src/analysis/axis_b_failures.py`)
    *   Classifies failures into Hallucination (subgoal invalid for observation) and Execution Mismatch (subgoal valid but not completed).
*   **Axis C: Gating Diagnostics** (`src/analysis/axis_c_gating.py`)
    *   Analyzes the timing and frequency of planner queries.
*   **Axis D: Representation** (`src/analysis/axis_d_representation.py`)
    *   Generates embeddings for unique subgoals using SentenceTransformers.
    *   Computes 2D projection (PCA) for visualization.

### 3. Visualization
`src/analysis/visualize.py`
Generates plots from the metrics produced by the analysis scripts.
*   Subgoal counts bar chart.
*   Failure mode distribution.
*   Gating query distribution histogram.
*   Subgoal embedding scatter plot.

### 4. Pipeline Driver
`src/analysis/run_pipeline.py`
Orchestrates the entire process:
1.  Collects traces for multiple tasks/noise levels.
2.  Runs all analysis scripts.
3.  Generates visualizations.

Usage:
```bash
python src/analysis/run_pipeline.py --id my_analysis --episodes 50
```

## Reproducibility
The analysis is deterministic given a seed. All scripts utilize the project's global seeding mechanism.
