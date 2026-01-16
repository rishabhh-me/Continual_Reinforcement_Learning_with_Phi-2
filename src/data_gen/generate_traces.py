import gymnasium as gym
import minigrid
import json
import os
import numpy as np
from tqdm import tqdm
from src.envs.wrappers import SubgoalWrapper
from src.data_gen.minigrid_solver import BFSController, get_low_level_plan
from src.data_gen.oracle import Oracle

def generate_dataset(num_episodes, env_id, output_path, start_seed=0):
    traces = []
    
    env = gym.make(env_id, render_mode="rgb_array")
    env = SubgoalWrapper(env)
    
    oracle = Oracle(env)
    
    pbar = tqdm(range(num_episodes), desc=f"Generating {output_path}")
    
    for i in pbar:
        seed = start_seed + i
        obs, info = env.reset(seed=seed)
        terminated = False
        truncated = False
        
        episode_trace = []
        
        while not terminated and not truncated:
            # 1. Get Canonical Subgoal
            sub_text, sub_tuple = oracle.get_next_subgoal()
            
            # 2. Plan Low-Level Actions
            actions = get_low_level_plan(env, sub_tuple)
            
            if not actions:
                # If no plan found, maybe we are exploring or stuck. 
                # For DoorKey, this shouldn't happen if Oracle is correct.
                # Just break to avoid infinite loop
                break
                
            # Execute Plan
            for action in actions:
                # Record State BEFORE action
                desc = env.get_text_description()
                negatives = oracle.get_negative_subgoals(sub_tuple)
                
                step_data = {
                    "state_text": desc,
                    "canonical_subgoal": sub_text,
                    "subgoal_tuple": sub_tuple,
                    "negatives": negatives,
                    "metadata": {
                        "seed": seed,
                        "env_id": env_id,
                        "map_id": 0 # TODO if multiple maps
                    }
                }
                episode_trace.append(step_data)
                
                # Step
                obs, reward, terminated, truncated, info = env.step(action)
                
                if terminated or truncated:
                    break
            
            # After executing the subgoal actions, we loop again to get the NEXT subgoal.
        
        # Add episode to traces
        traces.extend(episode_trace)

    # Save
    with open(output_path, 'w') as f:
        json.dump(traces, f, indent=2)
    
    print(f"Saved {len(traces)} steps to {output_path}")

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    # Train
    generate_dataset(num_episodes=50, env_id="MiniGrid-DoorKey-6x6-v0", output_path="data/expert_traces_train.json", start_seed=1000)
    
    # Val
    generate_dataset(num_episodes=10, env_id="MiniGrid-DoorKey-6x6-v0", output_path="data/expert_traces_val.json", start_seed=2000)
    
    # Test (Held Out)
    generate_dataset(num_episodes=10, env_id="MiniGrid-DoorKey-8x8-v0", output_path="data/expert_traces_test.json", start_seed=3000)
