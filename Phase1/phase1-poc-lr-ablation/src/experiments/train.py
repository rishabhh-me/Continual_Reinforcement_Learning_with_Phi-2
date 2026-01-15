import argparse
import os
import sys
import yaml
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Add src to path
sys.path.append(os.path.abspath("."))

from src.utils.logger import ExperimentLogger
from src.utils.subgoal_parser import parse_subgoal, Subgoal
from src.llm.planner import get_planner
from src.envs.wrappers import SubgoalWrapper
from src.ppo.sb3_agent import create_agent
from src.utils.seeding import set_global_seeds

class SubgoalUpdateCallback(BaseCallback):
    """
    Callback to handle subgoal updates during training.
    Checks for subgoal completion or initialization (NO_OP) and queries the planner.
    """
    def __init__(self, planner, verbose=0):
        super(SubgoalUpdateCallback, self).__init__(verbose)
        self.planner = planner
        self.subgoal_successes = 0
        self.total_subgoals = 0

    def _on_step(self) -> bool:
        # Iterate over all environments
        # self.locals['infos'] contains the info dicts for the current step
        # self.training_env is the VecEnv
        
        infos = self.locals['infos']
        
        for i, info in enumerate(infos):
            needs_update = False
            
            # Check for completion
            if info.get('subgoal_completed', False):
                if self.verbose > 0:
                    print(f"Env {i}: Subgoal completed!")
                self.subgoal_successes += 1
                needs_update = True
                
            # Check for NO_OP (initial state or reset)
            # We need to access the current subgoal ID from the env
            # Using env_method to get the attribute 'current_subgoal_id'
            # Note: access might be slow if done every step, but for POC it's fine.
            # Ideally we cache or track it.
            # But wait, we can just check if we "need update" based on info or first step.
            
            # For the very first step, current_subgoal_id might be 0.
            # We can try to get it.
            # However, getting attributes from VecEnv can be tricky if we don't know the wrapper depth.
            # Let's assume we rely on 'needs_update' which is set by completion.
            # BUT, initially it is 0.
            
            # Let's force an update if it's the first step (n_steps == 0) for that env?
            # SB3 doesn't track per-env steps easily in callback.
            
            # Let's peek at the env wrapper.
            # Assumes DummyVecEnv
            env_wrapper = self.training_env.envs[i]
            # Unwrap until we find SubgoalWrapper
            while hasattr(env_wrapper, 'env'):
                if isinstance(env_wrapper, SubgoalWrapper):
                    break
                env_wrapper = env_wrapper.env
            
            if isinstance(env_wrapper, SubgoalWrapper):
                if env_wrapper.current_subgoal_id == 0: # NO_OP
                    needs_update = True
            
            if needs_update:
                # Get text description
                # We can call get_text_description on the wrapper
                state_text = env_wrapper.get_text_description()
                
                # Query Planner
                subgoal_text = self.planner.generate_subgoal(state_text)
                
                # Parse
                subgoal_tuple, subgoal_id = parse_subgoal(subgoal_text)
                
                if self.verbose > 0:
                    print(f"Env {i}: New Subgoal: {subgoal_text} -> {subgoal_tuple}")
                
                # Set in Env
                env_wrapper.set_subgoal(subgoal_tuple, subgoal_id)
                self.total_subgoals += 1

        return True

def run_experiment(lr, exp_id, total_timesteps=50000):
    # Load default config
    with open("src/configs/default_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # Override Config
    config['rl']['learning_rate'] = lr
    config['experiment']['id'] = exp_id
    config['rl']['total_timesteps'] = total_timesteps
    
    print(f"=== Running Experiment: {exp_id} with LR={lr} ===")
    
    # Seeding
    set_global_seeds(config['experiment']['seed'])
    
    # Environment
    env_id = config['env']['id']
    def make_env():
        env = gym.make(env_id, render_mode="rgb_array")
        env = SubgoalWrapper(env)
        env = Monitor(env) # For SB3 logging
        return env
    
    # Create VecEnv
    env = DummyVecEnv([make_env])
    
    # Planner
    planner = get_planner(config)
    
    # Agent
    agent = create_agent(env, config)
    
    # Callback
    callback = SubgoalUpdateCallback(planner, verbose=1)
    
    # Train
    start_time = time.time()
    agent.learn(total_timesteps=config['rl']['total_timesteps'], callback=callback)
    end_time = time.time()
    
    print(f"Training finished in {end_time - start_time:.2f}s")
    print(f"Total Subgoals Issued: {callback.total_subgoals}")
    print(f"Total Subgoals Completed: {callback.subgoal_successes}")
    if callback.total_subgoals > 0:
        print(f"Subgoal Success Rate: {callback.subgoal_successes / callback.total_subgoals:.2f}")
    
    # Save Model
    save_path = f"experiments/runs/{exp_id}/model"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    agent.save(save_path)
    print(f"Model saved to {save_path}")
    
    return callback.subgoal_successes, callback.total_subgoals

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning Rate")
    parser.add_argument("--id", type=str, default="poc_experiment", help="Experiment ID")
    parser.add_argument("--steps", type=int, default=20000, help="Total Timesteps")
    args = parser.parse_args()
    
    run_experiment(args.lr, args.id, args.steps)
