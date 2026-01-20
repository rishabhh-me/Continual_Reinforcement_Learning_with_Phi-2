import argparse
import os
import sys
import subprocess
import time

sys.path.append(os.path.abspath("."))

def run_command(cmd):
    print(f"Running: {cmd}")
    subprocess.check_call(cmd, shell=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default="phase7_analysis")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--agent", type=str, default="scripted", help="Agent type (phi2, rehearsal, scripted)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to PPO model checkpoint")
    parser.add_argument("--mock", action="store_true", help="Use mock planner (overrides config)")

    args = parser.parse_args()

    # 1. Data Collection
    print("=== Step 1: Data Collection ===")

    tasks = ['MiniGrid-DoorKey-6x6-v0', 'MiniGrid-Unlock-v0']
    noise_levels = [0.0, 0.1]

    agent_arg = args.agent
    model_path_arg = f"--model_path {args.model_path}" if args.model_path else ""

    for task in tasks:
        for noise in noise_levels:
            cmd = (
                f"python src/analysis/collect_traces.py "
                f"--agent {agent_arg} "
                f"{model_path_arg} "
                f"--task {task} "
                f"--id {args.id} "
                f"--episodes {args.episodes} "
                f"--noise_type gaussian "
                f"--noise_level {noise}"
            )
            try:
                run_command(cmd)
            except Exception as e:
                print(f"Failed to run collection for {task} noise {noise}: {e}")

    trace_dir = f"analysis_output/traces/{args.id}"
    metrics_dir = f"analysis_output/metrics/{args.id}"
    figures_dir = f"analysis_output/figures/{args.id}"

    # 2. Analysis
    print("=== Step 2: Analysis ===")

    # Axis A
    run_command(f"python src/analysis/axis_a_subgoals.py --trace_dir {trace_dir} --output {metrics_dir}/axis_a_metrics.json")

    # Axis B
    run_command(f"python src/analysis/axis_b_failures.py --trace_dir {trace_dir} --output {metrics_dir}/axis_b_metrics.json")

    # Axis C
    run_command(f"python src/analysis/axis_c_gating.py --trace_dir {trace_dir} --output {metrics_dir}/axis_c_metrics.json")

    # Axis D
    run_command(f"python src/analysis/axis_d_representation.py --trace_dir {trace_dir} --output {metrics_dir}/axis_d_embeddings.json")

    # 3. Visualization
    print("=== Step 3: Visualization ===")
    run_command(f"python src/analysis/visualize.py --metrics_dir {metrics_dir} --output_dir {figures_dir}")

    print("=== Phase 7 Pipeline Complete ===")
    print(f"Results in {figures_dir}")

if __name__ == "__main__":
    main()
