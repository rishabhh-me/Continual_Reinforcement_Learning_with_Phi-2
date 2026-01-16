import argparse
import gymnasium as gym
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from src.envs.wrappers import SubgoalWrapper
from src.utils.subgoal_parser import parse_subgoal
from src.data_gen.minigrid_solver import get_low_level_plan

def run_evaluation(model, tokenizer, env_id, num_episodes=10):
    env = gym.make(env_id, render_mode="rgb_array")
    env = SubgoalWrapper(env)
    
    successes = 0
    total_steps = 0
    
    print(f"Evaluating on {env_id} with {num_episodes} episodes...")
    for i in tqdm(range(num_episodes)):
        obs, info = env.reset(seed=10000+i) # Deterministic evaluation
        terminated = False
        truncated = False
        step_count = 0
        
        while not terminated and not truncated:
            # Get State Text
            desc = env.get_text_description()
            
            # Query LLM
            prompt = (
                "Objective: Reach the goal.\n"
                "Rules:\n"
                "1. If you see a key and are carrying nothing, pick up the key.\n"
                "2. If you have the key and see a door, open the door.\n"
                "3. If the door is open, go to the goal.\n\n"
                f"Current State: {desc}\n"
                "Next Subgoal:"
            )
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=20, pad_token_id=tokenizer.eos_token_id)
            
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract subgoal
            if "Next Subgoal:" in output_text:
                subgoal_text = output_text.split("Next Subgoal:")[-1].strip().split("\n")[0]
            else:
                subgoal_text = output_text.replace(prompt, "").strip().split("\n")[0]
            
            # Parse
            subgoal_tuple, subgoal_id = parse_subgoal(subgoal_text)
            
            # Execute (using BFS Solver as Oracle Executor)
            actions = get_low_level_plan(env, subgoal_tuple)
            
            if not actions:
                # LLM gave invalid or impossible subgoal
                step_count += 1
                if step_count > 100: break
                continue
            
            for action in actions:
                obs, reward, terminated, truncated, info = env.step(action)
                step_count += 1
                if terminated or truncated: break
        
        if terminated and reward > 0:
            successes += 1
            
    success_rate = successes/num_episodes
    print(f"Success Rate: {success_rate}")
    return success_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--env_id", type=str, default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--load_in_4bit", action="store_true")
    
    args = parser.parse_args()
    
    bnb_config = None
    if args.load_in_4bit and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    )
    
    if args.adapter_path:
        print(f"Loading adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    run_evaluation(model, tokenizer, args.env_id, args.num_episodes)
