import argparse
import os
import sys
import yaml
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.abspath("."))

from src.utils.logger import ExperimentLogger
from src.utils.evaluation import evaluate_agent
from src.llm.planner import get_planner
from src.ppo.sb3_agent import create_agent
from src.envs.wrappers import SubgoalWrapper, FilterMissionWrapper
from src.envs.noise_wrappers import ObservationNoiseWrapper, TransitionNoiseWrapper, RewardNoiseWrapper
from src.utils.seeding import set_global_seeds

def run_robustness_experiment(args):
    # Load Config
    with open("src/configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Overrides
    config['rl']['learning_rate'] = args.lr
    config['experiment']['seed'] = args.seed
    config['experiment']['id'] = args.id

    set_global_seeds(args.seed)

    # Setup Planner
    use_subgoals = args.agent != 'flat'
    planner = None
    if use_subgoals:
        # Config Tweaks for Planner similar to train_continual.py
        if args.agent == 'scripted':
            config['llm']['mock_mode'] = True
        elif args.agent in ['phi2', 'rehearsal']:
            config['llm']['mock_mode'] = False
            if args.llm_model == 'frozen':
                config['llm']['use_lora'] = False
            elif args.llm_model == 'dpo':
                config['llm']['use_lora'] = True
                config['llm']['adapter_path'] = 'artifacts/phase2/dpo_lora/'
            elif args.llm_model == 'rlhf':
                config['llm']['use_lora'] = True
                config['llm']['adapter_path'] = 'artifacts/phase2/rlhf_lora/'

            if args.adapter_path:
                config['llm']['adapter_path'] = args.adapter_path

            config['llm']['prompt_mode'] = args.subgoal_mode

        planner = get_planner(config)

    # Prepare Save Path
    base_save_path = f"experiments/runs/{args.id}/seed_{args.seed}"
    os.makedirs(base_save_path, exist_ok=True)

    # Define Noise Config
    noise_config = {
        'obs_noise_type': args.obs_noise_type,
        'obs_intensity': args.obs_intensity,
        'transition_slip': args.transition_slip,
        'reward_noise': args.reward_noise
    }

    # Load Agent
    # If args.load_model_path is provided, load it.
    # Else, if transfer/training, maybe create new?
    # Usually robustness tests require a pretrained agent.

    env_id = args.task

    # Dummy Env for Agent Creation/Loading
    def make_env():
        # Apply noise wrappers here?
        # For loading, we just need compatible observation space.
        # SB3 checks observation space.
        # ObservationNoiseWrapper preserves shape.
        e = gym.make(env_id, render_mode="rgb_array")
        if use_subgoals:
             e = SubgoalWrapper(e, use_intrinsic_reward=False)
        else:
             e = FilterMissionWrapper(e)
        return e

    dummy_env = DummyVecEnv([make_env])

    if args.load_model_path:
        print(f"Loading model from {args.load_model_path}...")
        if args.agent == 'rehearsal':
            from src.ppo.rehearsal_agent import RehearsalPPO
            agent = RehearsalPPO.load(args.load_model_path, env=dummy_env)
        else:
            agent = PPO.load(args.load_model_path, env=dummy_env)
    else:
        print("No model path provided. Training from scratch or untrained evaluation.")
        agent = create_agent(dummy_env, config, agent_type=args.agent)

    # Freeze Features if requested (Representation Reuse)
    if args.freeze_features:
        print("Freezing feature extractor (Representation Reuse)...")
        for param in agent.policy.features_extractor.parameters():
            param.requires_grad = False

    # Mode 1: Evaluation (Robustness/Generalization)
    if args.mode == 'eval':
        print(f"Starting Evaluation on {env_id} with Noise: {noise_config}")

        # Generalization: Loop over seeds if provided?
        # Or just use args.seed.
        # If Generalization -> Unseen seeds. args.seed is the base.

        sr = evaluate_agent(
            agent,
            env_id,
            n_episodes=args.eval_episodes,
            seed=args.seed + 2000, # Different offset for robustness/gen
            use_subgoals=use_subgoals,
            planner=planner,
            noise_config=noise_config
        )
        print(f"Success Rate: {sr:.2f}")

        # Save Result
        result = {
            'task': env_id,
            'noise_type': args.obs_noise_type,
            'noise_intensity': args.obs_intensity,
            'transition_slip': args.transition_slip,
            'success_rate': sr
        }
        pd.DataFrame([result]).to_csv(f"{base_save_path}/robustness_result.csv", index=False)

    # Mode 2: Transfer (Fine-tuning)
    elif args.mode == 'transfer':
        print(f"Starting Transfer Training on {env_id}...")

        # Create Training Env with Noise (if any) and Wrappers
        def make_train_env():
            e = gym.make(env_id, render_mode="rgb_array")

            if use_subgoals:
                e = SubgoalWrapper(e, use_intrinsic_reward=True) # Enable intrinsic for training
            else:
                e = FilterMissionWrapper(e)

            # Apply Noise Wrappers (Outer)
            if args.obs_noise_type or args.obs_intensity > 0:
                e = ObservationNoiseWrapper(e, args.obs_noise_type, args.obs_intensity)
            if args.transition_slip > 0:
                e = TransitionNoiseWrapper(e, args.transition_slip)
            if args.reward_noise > 0:
                e = RewardNoiseWrapper(e, args.reward_noise)

            e = Monitor(e)
            return e

        train_env = DummyVecEnv([make_train_env])
        agent.set_env(train_env)

        # Train
        steps = args.steps
        agent.learn(total_timesteps=steps)

        # Save
        ckpt_path = f"{base_save_path}/transfer_{env_id}/model"
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        agent.save(ckpt_path)
        print(f"Transfer model saved to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, required=True, choices=['phi2', 'flat', 'scripted', 'rehearsal'])
    parser.add_argument("--mode", type=str, default='eval', choices=['eval', 'transfer'])
    parser.add_argument("--task", type=str, default='MiniGrid-DoorKey-8x8-v0')
    parser.add_argument("--load_model_path", type=str, default=None)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id", type=str, required=True)

    # Noise Params
    parser.add_argument("--obs_noise_type", type=str, default=None, choices=['gaussian', 'text_mask', 'text_replace'])
    parser.add_argument("--obs_intensity", type=float, default=0.0)
    parser.add_argument("--transition_slip", type=float, default=0.0)
    parser.add_argument("--reward_noise", type=float, default=0.0)

    # Representation Reuse
    parser.add_argument("--freeze_features", action="store_true", help="Freeze feature extractor weights.")

    # LLM Params
    parser.add_argument("--llm_model", type=str, default='dpo')
    parser.add_argument("--subgoal_mode", type=str, default='constrained')
    parser.add_argument("--adapter_path", type=str, default=None)

    args = parser.parse_args()
    run_robustness_experiment(args)
