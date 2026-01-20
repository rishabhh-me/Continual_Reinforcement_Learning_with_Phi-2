import argparse
import os
import sys
import yaml
import time
import numpy as np
import gymnasium as gym
import torch
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

sys.path.append(os.path.abspath("."))

from src.utils.logger import ExperimentLogger
from src.utils.subgoal_parser import parse_subgoal
from src.llm.planner import get_planner
from src.envs.wrappers import SubgoalWrapper, FilterMissionWrapper
from src.ppo.sb3_agent import create_agent
from src.utils.seeding import set_global_seeds
from src.utils.callbacks import SubgoalUpdateCallback
from src.data_gen.gen_continual_traces import generate_synthetic_data
from src.utils.evaluation import evaluate_agent

TASKS = [
    'MiniGrid-DoorKey-6x6-v0',
    'MiniGrid-DoorKey-8x8-v0',
    'MiniGrid-Unlock-v0',
    'MiniGrid-KeyCorridorS3R1-v0',
    'MiniGrid-ObstructedMaze-2Dlh-v0'
]

def run_continual_experiment(args):
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
        # Config Tweaks for Planner
        if args.agent == 'scripted':
            config['llm']['mock_mode'] = True
        elif args.agent in ['phi2', 'rehearsal']:
            config['llm']['mock_mode'] = False
            
            # Phase 4 Ablation Logic
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
            
            if args.finetune_mode == 'full':
                config['llm']['use_lora'] = False
            
            if args.model_path:
                config['llm']['model_name'] = args.model_path
            
            config['llm']['prompt_mode'] = args.subgoal_mode
        
        planner = get_planner(config)
    
    # Agent Init
    agent = None
    
    results = []
    base_save_path = f"experiments/runs/{args.id}/seed_{args.seed}"
    os.makedirs(base_save_path, exist_ok=True)
    
    global_subgoals = set()

    for task_idx, task_id in enumerate(TASKS):
        print(f"\n=== Starting Task {task_idx}: {task_id} ===")
        
        # Training Env
        def make_env():
            e = gym.make(task_id, render_mode="rgb_array")
            if use_subgoals:
                use_reward = bool(args.intrinsic_reward)
                e = SubgoalWrapper(e, use_intrinsic_reward=use_reward)
            else:
                e = FilterMissionWrapper(e)
            e = Monitor(e)
            return e
            
        env = DummyVecEnv([make_env])
        
        # Initialize Agent if first task
        if agent is None:
            agent = create_agent(env, config, agent_type=args.agent)
        else:
            agent.set_env(env)
            # Reset buffer to ensure we don't carry over stale data from previous task directly
            # (though RehearsalPPO stores it separately)
            # SB3 PPO reset() is handled internally? 
            # We assume agent.learn() calls collect_rollouts which resets buffer.
            pass

        # Phase 5: Load Synthetic Rehearsal Data
        if args.use_synthetic_rehearsal and args.agent == 'rehearsal':
            # Clear old data to prevent duplication as we reload all previous tasks
            agent.clear_rehearsal_data()
            
            for prev_idx in range(task_idx):
                prev_task_id = TASKS[prev_idx]
                data_path = f"{base_save_path}/task_{prev_idx}_{prev_task_id}/synthetic_data.pkl"
                if os.path.exists(data_path):
                    agent.load_rehearsal_data(data_path)
            
        # Callbacks
        callbacks = []
        subgoal_callback = None
        if use_subgoals:
            subgoal_callback = SubgoalUpdateCallback(planner, previously_seen_subgoals=global_subgoals, verbose=0)
            callbacks.append(subgoal_callback)
            
        # Train
        steps = args.steps_per_task
        print(f"Training for {steps} steps...")
        agent.learn(total_timesteps=steps, callback=callbacks)
        
        # Cache Rehearsal Data
        if args.agent == 'rehearsal':
            if args.use_synthetic_rehearsal:
                # Generate new synthetic data
                data_path = f"{base_save_path}/task_{task_idx}_{task_id}/synthetic_data.pkl"
                generate_synthetic_data(agent, env, steps=args.synthetic_steps, output_path=data_path, planner=planner)
            else:
                # Use default cache
                agent.cache_current_task_data()
            
        # Save
        ckpt_path = f"{base_save_path}/task_{task_idx}_{task_id}/model"
        agent.save(ckpt_path)
        print(f"Model saved to {ckpt_path}")

        # Metrics: Subgoal Reuse
        if subgoal_callback:
            global_subgoals.update(subgoal_callback.current_task_subgoals)
            
        # Evaluation Loop (Forgetting)
        print("Evaluating on all seen tasks...")
        current_results = {
            'train_task': task_id,
            'train_idx': task_idx,
        }

        if subgoal_callback:
            current_results['subgoal_reuse_count'] = subgoal_callback.reused_generation_count
            current_results['subgoal_unique_count'] = len(subgoal_callback.current_task_subgoals)
            current_results['total_subgoals_generated'] = subgoal_callback.total_subgoals
        
        for eval_idx, eval_task in enumerate(TASKS[:task_idx+1]):
            print(f"  Evaluating {eval_task}...")
            sr = evaluate_agent(
                agent, 
                eval_task, 
                n_episodes=args.eval_episodes, 
                seed=args.seed+1000+(task_idx*100), # Different seed offset per eval phase to avoid overfitting? Or fixed? User: "Fixed deterministic seeds (>=100)"
                use_subgoals=use_subgoals, 
                planner=planner
            )
            print(f"  Result {eval_task}: SR={sr:.2f}")
            current_results[f"eval_sr_{eval_task}"] = sr
            
        results.append(current_results)
        
        # Save Results DataFrame
        df = pd.DataFrame(results)
        df.to_csv(f"{base_save_path}/results.csv", index=False)
        print(f"Results updated: {base_save_path}/results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, required=True, choices=['phi2', 'flat', 'scripted', 'rehearsal'])
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps_per_task", type=int, default=100000)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--id", type=str, required=True)

    # Phase 4 Ablation Arguments
    parser.add_argument("--llm_model", type=str, choices=['frozen', 'dpo', 'rlhf'], default='dpo')
    parser.add_argument("--intrinsic_reward", type=int, default=1, help="1 to enable, 0 to disable")
    parser.add_argument("--subgoal_mode", type=str, choices=['constrained', 'freeform'], default='constrained')
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--finetune_mode", type=str, choices=['lora', 'full'], default='lora')
    parser.add_argument("--model_path", type=str, default=None)

    # Phase 5 Arguments
    parser.add_argument("--use_synthetic_rehearsal", action="store_true", help="Enable Phase 5 LLM-driven synthetic rehearsal")
    parser.add_argument("--synthetic_steps", type=int, default=10000, help="Number of synthetic steps to generate per task")
    
    args = parser.parse_args()
    run_continual_experiment(args)