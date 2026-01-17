import torch
from abc import ABC, abstractmethod
import os

class BasePlanner(ABC):
    @abstractmethod
    def generate_subgoal(self, state_description: str) -> str:
        pass

class MockPlanner(BasePlanner):
    """
    Deterministic mock planner for testing and CPU environments.
    """
    def __init__(self, env_id="MiniGrid-DoorKey-6x6-v0"):
        self.env_id = env_id

    def generate_subgoal(self, state_description: str) -> str:
        """
        Returns a canonical subgoal string based on heuristics.
        """
        desc = state_description.lower()
        
        # Parse State
        carrying = "nothing"
        if "carrying" in desc:
            # Example: "You are carrying a red key."
            parts = desc.split("carrying")
            if len(parts) > 1:
                carrying_part = parts[1].split(".")[0].strip()
                carrying = carrying_part
        
        # Check specific conditions
        has_key = "key" in carrying
        
        # Check what we see
        # Simple string matching on the description part
        # "In the room, you see: yellow key, yellow door, green goal."
        
        see_key = "key" in desc and "carrying a" not in desc 
        # Note: if we are carrying a key, "key" is in desc (in the carrying part).
        # But we want to know if we see *another* key or the same key on the floor?
        # Usually description separates "You are carrying..." and "You see...".
        # Let's assume strict parsing isn't needed if we trust the context.
        # But wait, if I carry a key, "key" is in the string.
        # So "see_key" might be true if I carry it?
        # Let's check if "key" appears in the "see" section.
        see_section = ""
        if "see:" in desc:
            see_section = desc.split("see:")[1]
        
        see_key_in_room = "key" in see_section
        see_ball_in_room = "ball" in see_section
        
        # Door logic
        # "yellow door" -> usually closed. "open yellow door" -> open.
        # But simply checking "door" in see_section tells us there is a door.
        see_door = "door" in see_section
        is_door_open = "open" in see_section and "door" in see_section
        is_door_closed = see_door and not is_door_open

        # Heuristics
        
        # 1. Open Door if we have key and see closed door
        if has_key and is_door_closed:
            if "yellow" in see_section: return "Open the yellow door"
            if "red" in see_section: return "Open the red door"
            if "green" in see_section: return "Open the green door"
            if "blue" in see_section: return "Open the blue door"
            return "Open the door"

        # 2. Pick up key if we see it and carry nothing
        if see_key_in_room and "nothing" in carrying:
            if "yellow" in see_section: return "Pick up the yellow key"
            if "red" in see_section: return "Pick up the red key"
            if "green" in see_section: return "Pick up the green key"
            if "blue" in see_section: return "Pick up the blue key"
            return "Pick up the key"

        # 3. Pick up ball if we see it and carry nothing (Goal for KeyCorridor/ObstructedMaze)
        if see_ball_in_room and "nothing" in carrying:
             if "yellow" in see_section: return "Pick up the yellow ball"
             if "purple" in see_section: return "Pick up the purple ball"
             if "green" in see_section: return "Pick up the green ball"
             return "Pick up the ball"

        # 4. Go to goal
        if "goal" in see_section:
            return "Go to the goal"
            
        # 5. Explore if nothing else triggers
        return "Explore"

class Phi2Planner(BasePlanner):
    def __init__(self, model_name="microsoft/phi-2", load_in_4bit=True, use_lora=False, adapter_path=None, prompt_mode="constrained", device_map="auto", temperature=0.1, max_new_tokens=50):
        if not torch.cuda.is_available():
             raise RuntimeError("Phi2Planner requires CUDA. Use MockPlanner instead.")

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.prompt_mode = prompt_mode

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        print(f"Loading Base Model: {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )

        if use_lora:
            if adapter_path is None:
                adapter_path = "artifacts/phase2/dpo_lora/" # Default Phase 3 path

            if os.path.exists(adapter_path):
                print(f"Loading LoRA Adapters from {adapter_path}...")
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
                self.model.requires_grad_(False) # Freeze
            else:
                print(f"Warning: LoRA path {adapter_path} not found. Running with base model.")

    def generate_subgoal(self, state_description: str) -> str:
        # Prompt construction
        if self.prompt_mode == "freeform":
            prompt = (
                f"Current State: {state_description}\n"
                "Next Subgoal:"
            )
        else:
            # Constrained (default)
            prompt = (
                "Objective: Reach the goal.\n"
                "Rules:\n"
                "1. If you see a key and are carrying nothing, pick up the key.\n"
                "2. If you have the key and see a door, open the door.\n"
                "3. If the door is open, go to the goal.\n\n"
                "Example 1:\n"
                "Current State: You are carrying nothing. In the room, you see: yellow key, yellow door, green goal.\n"
                "Next Subgoal: Pick up the yellow key\n\n"
                "Example 2:\n"
                "Current State: You are carrying a yellow key. In the room, you see: yellow door, green goal.\n"
                "Next Subgoal: Open the yellow door\n\n"
                f"Current State: {state_description}\n"
                "Next Subgoal:"
            )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=(self.temperature > 0),
                pad_token_id=self.tokenizer.eos_token_id
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the part after "Next Subgoal:"
        if "Next Subgoal:" in text:
            subgoal = text.split("Next Subgoal:")[-1].strip().split("\n")[0]
        else:
            subgoal = text.strip()

        return subgoal

def get_planner(config):
    mock_mode = config['llm'].get('mock_mode', 'auto')
    
    use_mock = False
    if mock_mode == True:
        use_mock = True
    elif mock_mode == 'auto':
        if not torch.cuda.is_available():
            use_mock = True

    if use_mock:
        print("Using MockPlanner (CPU/Testing Mode)")
        return MockPlanner(env_id=config['env']['id'])
    else:
        print("Using Phi2Planner (GPU Mode)")
        return Phi2Planner(
            model_name=config['llm']['model_name'],
            load_in_4bit=config['llm']['load_in_4bit'],
            use_lora=config['llm']['use_lora'],
            adapter_path=config['llm'].get('adapter_path', None),
            prompt_mode=config['llm'].get('prompt_mode', 'constrained'),
            temperature=config['llm']['temperature'],
            max_new_tokens=config['llm']['max_new_tokens']
        )