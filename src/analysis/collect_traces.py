import argparse
import os
import sys
import yaml
import numpy as np
import gymnasium as gym
import json
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Add project root to path
sys.path.append(os.path.abspath("."))

from src.utils.logger import ExperimentLogger
from src.llm.planner import get_planner
from src.envs.wrappers import SubgoalWrapper, FilterMissionWrapper
from src.envs.noise_wrappers import ObservationNoiseWrapper, TransitionNoiseWrapper, RewardNoiseWrapper
from src.ppo.sb3_agent import create_agent
from src.utils.seeding import set_global_seeds
from src.utils.subgoal_parser import parse_subgoal
from src.analysis.utils import ensure_dir

def collect_traces(
    agent,
    env_id,
    n_episodes,
    seed,
    use_subgoals,
    planner=None,
    noise_wrappers=None,
    output_file="traces.jsonl",
    always_replan=False,
    blocked_subgoals=None
):
    """
    Runs episodes and logs detailed traces to a JSONL file.
    """
    print(f"Collecting traces for {env_id} into {output_file}...")
    ensure_dir(os.path.dirname(output_file))

    if blocked_subgoals is None:
        blocked_subgoals = []

    # Setup Environment
    env = gym.make(env_id, render_mode="rgb_array")

    if use_subgoals:
        env = SubgoalWrapper(env, use_intrinsic_reward=False)
    else:
        env = FilterMissionWrapper(env)

    if noise_wrappers:
        for WrapperClass, kwargs in noise_wrappers:
            env = WrapperClass(env, **kwargs)

    # Open file
    with open(output_file, "w") as f_out:

        for ep_idx in range(n_episodes):
            obs, info = env.reset(seed=seed + ep_idx)

            # Initial Subgoal
            current_subgoal_text = "None"
            current_subgoal_tuple = ("no_op", "none", "none")

            def get_and_set_subgoal(current_obs_desc):
                text = planner.generate_subgoal(current_obs_desc)

                # Causal Ablation: Block specific subgoals
                for blocked in blocked_subgoals:
                    if blocked.lower() in text.lower():
                        # Simple override strategy
                        text = "Explore"
                        break

                tup, sid = parse_subgoal(text)
                if hasattr(env, 'set_subgoal'):
                    env.set_subgoal(tup, sid)

                # Update observation
                # Note: 'obs' variable in the loop needs to be updated.
                # However, set_subgoal updates internal state.
                # The agent will see it next step or if we modify obs here.
                # We should return the ID so we can update obs['subgoal']
                return text, tup, sid

            if use_subgoals and planner:
                 desc = getattr(env, 'get_text_description', lambda: "")()
                 if desc:
                     current_subgoal_text, current_subgoal_tuple, sub_id = get_and_set_subgoal(desc)
                     if 'subgoal' in obs:
                         obs['subgoal'] = np.array([sub_id], dtype=np.int32)

            done = False
            truncated = False
            episode_reward = 0
            step_count = 0

            episode_trace = {
                "episode_id": ep_idx,
                "seed": seed + ep_idx,
                "steps": []
            }

            while not (done or truncated):
                step_data = {
                    "step": step_count,
                    "obs_desc": getattr(env, 'get_text_description', lambda: "")(),
                    "subgoal_text": current_subgoal_text,
                    "subgoal_tuple": list(current_subgoal_tuple),
                    "current_subgoal_id": int(getattr(env, 'current_subgoal_id', 0)),
                }

                # Action
                action, _ = agent.predict(obs, deterministic=True)
                step_data["action"] = int(action)

                # Step
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1

                step_data["reward"] = float(reward)
                step_data["done"] = done or truncated
                step_data["info"] = {k: str(v) for k,v in info.items()}

                # Subgoal Update Logic
                replan_occurred = False
                if use_subgoals and planner:
                    completed = info.get('subgoal_completed', False)
                    current_id = getattr(env, 'current_subgoal_id', 0)

                    # Counterfactual: always_replan
                    if completed or current_id == 0 or always_replan:
                         replan_occurred = True
                         desc = getattr(env, 'get_text_description', lambda: "")()
                         current_subgoal_text, current_subgoal_tuple, sub_id = get_and_set_subgoal(desc)

                         if 'subgoal' in obs:
                             obs['subgoal'][0] = sub_id

                step_data["replan"] = replan_occurred
                episode_trace["steps"].append(step_data)

            episode_trace["total_reward"] = float(episode_reward)
            episode_trace["success"] = 1 if episode_reward > 0 else 0

            f_out.write(json.dumps(episode_trace) + "\n")
            f_out.flush()

    env.close()
    print(f"Finished collecting {n_episodes} episodes.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, required=True, choices=['phi2', 'flat', 'scripted', 'rehearsal'])
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--noise_type", type=str, default='none')
    parser.add_argument("--noise_level", type=float, default=0.0)

    # New Arguments for Phase 7 Ablations
    parser.add_argument("--always_replan", action="store_true", help="Force planner query at every step")
    parser.add_argument("--blocked_subgoals", type=str, default="", help="Comma-separated list of subgoal strings to block")

    args = parser.parse_args()

    blocked_list = [s.strip() for s in args.blocked_subgoals.split(",")] if args.blocked_subgoals else []

    # Load Config
    with open("src/configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Overrides
    config['experiment']['seed'] = args.seed
    set_global_seeds(args.seed)

    # Setup Planner
    use_subgoals = args.agent != 'flat'
    planner = None
    if use_subgoals:
        if args.agent == 'scripted':
            config['llm']['mock_mode'] = True
        elif args.agent in ['phi2', 'rehearsal']:
             config['llm']['mock_mode'] = False

        planner = get_planner(config)

    # Load Agent
    dummy_env_instance = gym.make(args.task, render_mode="rgb_array")
    if use_subgoals:
        dummy_env_instance = SubgoalWrapper(dummy_env_instance)
    else:
        dummy_env_instance = FilterMissionWrapper(dummy_env_instance)

    dummy_vec_env = DummyVecEnv([lambda: dummy_env_instance])

    agent = None
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        if args.agent == 'rehearsal':
            from src.ppo.rehearsal_agent import RehearsalPPO
            agent = RehearsalPPO.load(args.model_path, env=dummy_vec_env)
        else:
            agent = PPO.load(args.model_path, env=dummy_vec_env)
    else:
        print("Warning: Model path not found or not provided. Using RANDOM initialized agent.")
        agent = create_agent(dummy_vec_env, config, agent_type=args.agent)

    # Setup Noise
    noise_wrappers = []
    if args.noise_type == 'text_mask':
        noise_wrappers.append((ObservationNoiseWrapper, {'text_noise_level': args.noise_level, 'text_noise_type': 'mask'}))
    elif args.noise_type == 'text_replace':
            noise_wrappers.append((ObservationNoiseWrapper, {'text_noise_level': args.noise_level, 'text_noise_type': 'replace'}))
    elif args.noise_type == 'gaussian':
        noise_wrappers.append((ObservationNoiseWrapper, {'image_noise_level': args.noise_level}))
    elif args.noise_type == 'transition':
        noise_wrappers.append((TransitionNoiseWrapper, {'transition_noise_level': args.noise_level}))
    elif args.noise_type == 'reward':
        noise_wrappers.append((RewardNoiseWrapper, {'reward_noise_level': args.noise_level}))

    output_path = f"analysis_output/traces/{args.id}/{args.task}_noise_{args.noise_type}_{args.noise_level}.jsonl"

    collect_traces(
        agent=agent,
        env_id=args.task,
        n_episodes=args.episodes,
        seed=args.seed,
        use_subgoals=use_subgoals,
        planner=planner,
        noise_wrappers=noise_wrappers,
        output_file=output_path,
        always_replan=args.always_replan,
        blocked_subgoals=blocked_list
    )

if __name__ == "__main__":
    main()
