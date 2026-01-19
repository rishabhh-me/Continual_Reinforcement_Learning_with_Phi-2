import pickle
import os
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.buffers import RolloutBuffer, DictRolloutBuffer
from gymnasium import spaces
from copy import deepcopy

from src.utils.callbacks import SubgoalUpdateCallback

def generate_synthetic_data(agent, env, steps, output_path, planner=None):
    """
    Generates synthetic data using the current agent and environment.

    Args:
        agent: The PPO/RehearsalPPO agent.
        env: The VecEnv to collect data from.
        steps: Number of steps to collect.
        output_path: Path to save the pickle file.
        planner: (Optional) Planner instance for SubgoalUpdateCallback.
    """
    print(f"Generating {steps} steps of synthetic data...")

    # Temporarily switch env
    # Note: agent.set_env() might wrap the env (e.g. VecTransposeImage).
    # We must do this BEFORE creating the buffer so we get the correct observation space.
    original_env = agent.env
    agent.set_env(env)

    current_env = agent.env

    # Choose Buffer Class
    if isinstance(current_env.observation_space, spaces.Dict):
        buffer_cls = DictRolloutBuffer
    else:
        buffer_cls = RolloutBuffer

    # Create a temporary buffer
    buffer = buffer_cls(
        buffer_size=steps,
        observation_space=current_env.observation_space,
        action_space=current_env.action_space,
        device=agent.device,
        gamma=agent.gamma,
        gae_lambda=agent.gae_lambda,
        n_envs=current_env.num_envs
    )

    # Ensure _last_obs is set (collect_rollouts needs it)
    # Must use current_env (which might be wrapped) to get correct observation shape
    if agent._last_obs is None:
        agent._last_obs = current_env.reset()

    # Callback
    callback = None
    if planner:
        # Pass empty set for metrics to avoid polluting global metrics during generation?
        # Or just let it run. The callback expects a planner.
        # We use a new instance to handle subgoal updates.
        callback = SubgoalUpdateCallback(planner, verbose=0)
        # Callbacks must be initialized with the model
        callback.init_callback(agent)

    # Collect
    agent.collect_rollouts(
        current_env,
        callback=callback,
        rollout_buffer=buffer,
        n_rollout_steps=steps
    )

    # Restore env
    if original_env is not None:
        agent.set_env(original_env)
        # Note: This resets original_env. If we want to preserve state, it's hard.
        # But usually generation happens after training, so reset is fine.

    # Extract data
    data = {
        'observations': deepcopy(buffer.observations),
        'actions': deepcopy(buffer.actions),
        'old_log_prob': deepcopy(buffer.log_probs),
        'advantages': deepcopy(buffer.advantages),
        'returns': deepcopy(buffer.returns),
        'values': deepcopy(buffer.values)
    }

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"Saved synthetic data to {output_path}")
