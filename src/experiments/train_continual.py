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
from src.envs.wrappers import SubgoalWrapper
from src.ppo.sb3_agent import create_agent
from src.utils.seeding import set_global_seeds
from src.utils.callbacks import SubgoalUpdateCallback

TASKS = [
    'MiniGrid-DoorKey-6x6-v0',
    'MiniGrid-DoorKey-8x8-v0',
    'MiniGrid-Unlock-v0',
    'MiniGrid-KeyCorridorS3R1-v0',
    'MiniGrid-ObstructedMaze-2Dlh-v0'
]

class FilterMissionWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Filter out mission which is Text space and breaks SB3 DummyVecEnv
        new_spaces = {k: v for k, v in env.observation_space.spaces.items() if k != 'mission'}
        self.observation_space = gym.spaces.Dict(new_spaces)
    
    def observation(self, obs):
        return {k: v for k, v in obs.items() if k != 'mission'}

def evaluate_agent(agent, env_id, n_episodes, seed, use_subgoals, planner=None):
    env = gym.make(env_id, render_mode="rgb_array")
    if use_subgoals:
        env = SubgoalWrapper(env)
    else:
        env = FilterMissionWrapper(env)
    
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
             obs['subgoal'] = np.array([sub_id], dtype=np.int32)
             
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if use_subgoals and planner:
                if info.get('subgoal_completed', False) or env.current_subgoal_id == 0:
                     desc = env.get_text_description()
                     sub_text = planner.generate_subgoal(desc)
                     sub_tuple, sub_id = parse_subgoal(sub_text)
                     env.set_subgoal(sub_tuple, sub_id)
                     obs['subgoal'][0] = sub_id

        rewards.append(episode_reward)
        if episode_reward > 0: successes += 1
    
    return successes / n_episodes

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
            config['llm']['use_lora'] = True # Phase 3 requirement
        
        planner = get_planner(config)
    
    # Agent Init
    agent = None
    
    results = []
    base_save_path = f"experiments/runs/{args.id}/seed_{args.seed}"
    os.makedirs(base_save_path, exist_ok=True)
    
    for task_idx, task_id in enumerate(TASKS):
        print(f"\n=== Starting Task {task_idx}: {task_id} ===")
        
        # Training Env
        def make_env():
            e = gym.make(task_id, render_mode="rgb_array")
            if use_subgoals:
                e = SubgoalWrapper(e)
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
            
        # Callbacks
        callbacks = []
        if use_subgoals:
            callbacks.append(SubgoalUpdateCallback(planner, verbose=0))
            
        # Train
        steps = args.steps_per_task
        print(f"Training for {steps} steps...")
        agent.learn(total_timesteps=steps, callback=callbacks)
        
        # Cache Rehearsal Data
        if args.agent == 'rehearsal':
            agent.cache_current_task_data()
            
        # Save
        ckpt_path = f"{base_save_path}/task_{task_idx}_{task_id}/model"
        agent.save(ckpt_path)
        print(f"Model saved to {ckpt_path}")
        
        # Evaluation Loop (Forgetting)
        print("Evaluating on all seen tasks...")
        current_results = {
            'train_task': task_id,
            'train_idx': task_idx,
        }
        
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
    
    args = parser.parse_args()
    run_continual_experiment(args)
