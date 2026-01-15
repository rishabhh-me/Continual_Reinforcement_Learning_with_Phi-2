# Phase 1 POC Report: Learning Rate Comparison

## Overview
This report compares the performance of the Subgoal-Conditioned PPO Executor with two different learning rates: `1e-4` and `3e-4` (default). The experiments were conducted using the `MockPlanner` in the `MiniGrid-DoorKey-6x6-v0` environment over 20,480 timesteps.

## Methodology
- **Environment**: `MiniGrid-DoorKey-6x6-v0` with `SubgoalWrapper`.
- **Agent**: PPO with `MiniGridFeaturesExtractor` (CNN + Subgoal Embedding).
- **Planner**: `MockPlanner` (deterministic rule-based).
- **Training Duration**: 20,480 steps per run.
- **Seeds**: Fixed global seed (42).

## Results

| Metric | LR = 1e-4 | LR = 3e-4 (Default) |
| :--- | :--- | :--- |
| **Subgoal Success Rate** | 53% (62/118) | 53% (62/117) |
| **Episode Reward Mean** | 1.08 | 1.09 |
| **Value Loss (Final)** | ~0.0232 | ~0.00398 |
| **Approx KL (Final)** | ~0.0065 | ~0.0161 |
| **Clip Fraction (Final)** | ~0.051 | ~0.166 |
| **Policy Gradient Loss** | -0.003 | -0.015 |

## Analysis
1.  **Task Performance**: Both learning rates achieved nearly identical task performance in terms of subgoal completion rate (53%) and average episode reward (~1.09). This suggests that for this short training duration and specific task, the agent is robust to this range of learning rates.
2.  **Value Function**: The run with `LR = 3e-4` achieved a significantly lower value loss (`0.004` vs `0.023`). This indicates that the critic network converged faster or more accurately to the true value function with the higher learning rate.
3.  **Policy Updates**: The `LR = 3e-4` run showed higher `approx_kl` and `clip_fraction`, indicating more aggressive policy updates. While this can sometimes lead to instability, in this case, it did not degrade performance and likely contributed to the better value function learning.

## Conclusion & Recommendation
Based on the lower value loss and similar/slightly better reward performance, the default learning rate of **3e-4** (or `2.5e-4` as suggested in the plan) appears to be more effective for this specific setup than `1e-4`. The aggressive updates did not cause instability, and the critic learned better.

For future long-term training, `3e-4` is recommended, though `1e-4` remains a safe, stable alternative if instability is observed in more complex environments.
