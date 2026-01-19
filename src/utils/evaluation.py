import gymnasium as gym
import numpy as np
from src.envs.wrappers import SubgoalWrapper, FilterMissionWrapper
from src.envs.noise_wrappers import ObservationNoiseWrapper, TransitionNoiseWrapper, RewardNoiseWrapper
from src.utils.subgoal_parser import parse_subgoal

def evaluate_agent(agent, env_id, n_episodes, seed, use_subgoals, planner=None, noise_config=None):
    """
    Evaluates the agent on a specific task with optional noise.

    Args:
        agent: The PPO/Rehearsal agent.
        env_id (str): Gym environment ID.
        n_episodes (int): Number of episodes to run.
        seed (int): Base seed for evaluation.
        use_subgoals (bool): Whether to use subgoals (hierarchical).
        planner (Planner, optional): The LLM planner.
        noise_config (dict, optional): Dict containing noise parameters:
            - obs_noise_type (str): 'gaussian', 'text_mask', etc.
            - obs_intensity (float)
            - transition_slip (float)
            - reward_noise (float)
    """
    env = gym.make(env_id, render_mode="rgb_array")

    if use_subgoals:
        # Disable intrinsic reward during evaluation to measure true task success
        env = SubgoalWrapper(env, use_intrinsic_reward=False)
    else:
        env = FilterMissionWrapper(env)

    # Apply Noise Wrappers (Outer, so they can affect SubgoalWrapper's output/text)
    if noise_config:
        if noise_config.get('obs_noise_type') or noise_config.get('obs_intensity', 0) > 0:
            env = ObservationNoiseWrapper(
                env,
                noise_type=noise_config.get('obs_noise_type', 'gaussian'),
                intensity=noise_config.get('obs_intensity', 0.0)
            )
        if noise_config.get('transition_slip', 0) > 0:
            env = TransitionNoiseWrapper(env, slip_prob=noise_config['transition_slip'])
        if noise_config.get('reward_noise', 0) > 0:
            env = RewardNoiseWrapper(env, noise_std=noise_config['reward_noise'])

    successes = 0
    rewards = []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed+i)

        # Initial Subgoal
        if use_subgoals and planner:
             desc = env.get_text_description()
             sub_text = planner.generate_subgoal(desc)
             sub_tuple, sub_id = parse_subgoal(sub_text)
             env.set_subgoal(sub_tuple, sub_id)
             if 'subgoal' in obs:
                 # Ensure subgoal is in obs (SubgoalWrapper handles it, but we update it)
                 obs['subgoal'] = np.array([sub_id], dtype=np.int32)

        done = False
        truncated = False
        episode_reward = 0

        while not (done or truncated):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            if use_subgoals and planner:
                if info.get('subgoal_completed', False) or getattr(env, 'current_subgoal_id', 0) == 0:
                     desc = env.get_text_description()
                     sub_text = planner.generate_subgoal(desc)
                     sub_tuple, sub_id = parse_subgoal(sub_text)
                     env.set_subgoal(sub_tuple, sub_id)
                     if 'subgoal' in obs:
                         obs['subgoal'][0] = sub_id

        rewards.append(episode_reward)
        if episode_reward > 0: successes += 1

    env.close()
    return successes / n_episodes
