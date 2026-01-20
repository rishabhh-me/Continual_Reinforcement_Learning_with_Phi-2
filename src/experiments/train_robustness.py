import argparse
import os
import sys
import yaml
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.abspath("."))

from src.utils.logger import ExperimentLogger
from src.llm.planner import get_planner
from src.envs.wrappers import SubgoalWrapper, FilterMissionWrapper
from src.envs.noise_wrappers import ObservationNoiseWrapper, TransitionNoiseWrapper, RewardNoiseWrapper
from src.ppo.sb3_agent import create_agent
from src.utils.seeding import set_global_seeds
from src.utils.callbacks import SubgoalUpdateCallback
from src.utils.evaluation import evaluate_agent

def run_robustness(args):
    # Load Config
    with open("src/configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Overrides
    config['experiment']['seed'] = args.seed
    config['experiment']['id'] = args.id
    set_global_seeds(args.seed)

    # Setup Planner
    use_subgoals = args.agent != 'flat'
    planner = None
    if use_subgoals:
        if args.agent == 'scripted':
            config['llm']['mock_mode'] = True
        elif args.agent in ['phi2', 'rehearsal']:
             config['llm']['mock_mode'] = False

             if args.llm_model == 'frozen':
                config['llm']['use_lora'] = False
             elif args.llm_model == 'dpo':
                config['llm']['use_lora'] = True
                config['llm']['adapter_path'] = 'artifacts/phase2/dpo_lora/'

             if args.adapter_path:
                 config['llm']['adapter_path'] = args.adapter_path

             config['llm']['prompt_mode'] = args.subgoal_mode

        planner = get_planner(config)

    # Load Agent
    # We create a dummy env to initialize/load the agent
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
            # RehearsalPPO loading might need specific handling if custom objects
            agent = RehearsalPPO.load(args.model_path, env=dummy_vec_env)
        else:
            agent = PPO.load(args.model_path, env=dummy_vec_env)
    else:
        if args.model_path:
            print(f"Warning: Model path {args.model_path} not found. Creating new agent.")
        else:
            print("Creating new agent (Scratch)")
        agent = create_agent(dummy_vec_env, config, agent_type=args.agent)

    # Mode: Evaluate Robustness
    if args.mode == 'evaluate':
        print(f"Evaluating robustness on {args.task}")
        print(f"Noise: {args.noise_type}, Level: {args.noise_level}")

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

        sr = evaluate_agent(
            agent,
            args.task,
            n_episodes=args.eval_episodes,
            seed=args.seed + 1000,
            use_subgoals=use_subgoals,
            planner=planner,
            noise_wrappers=noise_wrappers
        )
        print(f"Success Rate: {sr}")

        # Log to file
        os.makedirs(f"experiments/runs/{args.id}", exist_ok=True)
        with open(f"experiments/runs/{args.id}/robustness.txt", "a") as f:
            f.write(f"{args.task},{args.noise_type},{args.noise_level},{sr}\n")

    # Mode: Transfer (Train)
    elif args.mode == 'transfer':
        print(f"Transfer learning on {args.task}")

        if args.freeze_features:
            print("Freezing feature extractor...")
            # PPO policy feature extractor
            for param in agent.policy.features_extractor.parameters():
                param.requires_grad = False

        # Training Loop
        def make_env():
            e = gym.make(args.task, render_mode="rgb_array")
            if use_subgoals:
                e = SubgoalWrapper(e, use_intrinsic_reward=True)
            else:
                e = FilterMissionWrapper(e)

            e = Monitor(e)
            return e

        env = DummyVecEnv([make_env])
        agent.set_env(env)

        callbacks = []
        if use_subgoals:
            subgoal_callback = SubgoalUpdateCallback(planner, verbose=0)
            callbacks.append(subgoal_callback)

        print(f"Training for {args.steps} steps...")
        agent.learn(total_timesteps=args.steps, callback=callbacks)

        # Save
        save_path = f"experiments/runs/{args.id}/transfer_{args.task}"
        agent.save(save_path)
        print(f"Saved to {save_path}")

        # Evaluate after transfer
        sr = evaluate_agent(
            agent,
            args.task,
            n_episodes=args.eval_episodes,
            seed=args.seed + 2000,
            use_subgoals=use_subgoals,
            planner=planner
        )
        print(f"Post-Transfer Success Rate: {sr}")

        # Log
        os.makedirs(f"experiments/runs/{args.id}", exist_ok=True)
        with open(f"experiments/runs/{args.id}/transfer.txt", "a") as f:
            f.write(f"{args.task},{args.freeze_features},{sr}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=['evaluate', 'transfer'])
    parser.add_argument("--agent", type=str, required=True, choices=['phi2', 'flat', 'scripted', 'rehearsal'])
    parser.add_argument("--model_path", type=str, default=None, help="Path to pretrained PPO model")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id", type=str, required=True)

    # Noise Args
    parser.add_argument("--noise_type", type=str, default='none', choices=['none', 'text_mask', 'text_replace', 'gaussian', 'transition', 'reward'])
    parser.add_argument("--noise_level", type=float, default=0.0)

    # Transfer Args
    parser.add_argument("--freeze_features", action="store_true")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--eval_episodes", type=int, default=50)

    # LLM Args
    parser.add_argument("--llm_model", type=str, choices=['frozen', 'dpo'], default='dpo')
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--subgoal_mode", type=str, default='constrained')

    args = parser.parse_args()
    run_robustness(args)
