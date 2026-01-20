import gymnasium as gym
import numpy as np
from src.envs.wrappers import SubgoalWrapper, FilterMissionWrapper
from src.utils.subgoal_parser import parse_subgoal

def evaluate_agent(agent, env_id, n_episodes, seed, use_subgoals, planner=None, noise_wrappers=None):
    """
    Evaluates the agent on a specific environment.

    Args:
        agent: The PPO agent.
        env_id: MiniGrid environment ID.
        n_episodes: Number of episodes to run.
        seed: Random seed.
        use_subgoals: Whether to use subgoals (and Planner).
        planner: The LLM planner instance (optional).
        noise_wrappers: List of (WrapperClass, kwargs) to apply to the environment.
    """
    env = gym.make(env_id, render_mode="rgb_array")

    if use_subgoals:
        # Disable intrinsic reward during evaluation to measure true task success
        env = SubgoalWrapper(env, use_intrinsic_reward=False)
    else:
        env = FilterMissionWrapper(env)

    if noise_wrappers:
        for WrapperClass, kwargs in noise_wrappers:
            env = WrapperClass(env, **kwargs)

    successes = 0
    rewards = []

    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed+i)

        # Initial Subgoal
        if use_subgoals and planner:
             # robustly get text description
             desc = getattr(env, 'get_text_description', lambda: "")()
             if not desc:
                  # Try to dig? No, wrappers should delegate.
                  pass

             sub_text = planner.generate_subgoal(desc)
             sub_tuple, sub_id = parse_subgoal(sub_text)

             if hasattr(env, 'set_subgoal'):
                 env.set_subgoal(sub_tuple, sub_id)

             if 'subgoal' in obs:
                 obs['subgoal'] = np.array([sub_id], dtype=np.int32)

        done = False
        truncated = False
        episode_reward = 0

        while not (done or truncated):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

            if use_subgoals and planner:
                completed = info.get('subgoal_completed', False)
                current_id = getattr(env, 'current_subgoal_id', 0)

                if completed or current_id == 0:
                     desc = getattr(env, 'get_text_description', lambda: "")()
                     sub_text = planner.generate_subgoal(desc)
                     sub_tuple, sub_id = parse_subgoal(sub_text)

                     if hasattr(env, 'set_subgoal'):
                         env.set_subgoal(sub_tuple, sub_id)

                     if 'subgoal' in obs:
                         obs['subgoal'][0] = sub_id

        rewards.append(episode_reward)
        if episode_reward > 0: successes += 1

    return successes / n_episodes
