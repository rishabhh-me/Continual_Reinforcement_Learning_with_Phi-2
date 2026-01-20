from stable_baselines3.common.callbacks import BaseCallback
from src.utils.subgoal_parser import parse_subgoal
from src.envs.wrappers import SubgoalWrapper

class SubgoalUpdateCallback(BaseCallback):
    def __init__(self, planner, previously_seen_subgoals=None, verbose=0):
        super(SubgoalUpdateCallback, self).__init__(verbose)
        self.planner = planner
        self.subgoal_successes = 0
        self.total_subgoals = 0
        
        # Phase 5: Subgoal Reuse Tracking
        # previously_seen_subgoals: set of canonical subgoal tuples seen in prior tasks
        self.previously_seen_subgoals = previously_seen_subgoals if previously_seen_subgoals is not None else set()
        self.current_task_subgoals = set() # Unique subgoals seen in this task
        self.reused_generation_count = 0 # How many times we generated a subgoal that was seen in previous tasks

    def _on_step(self) -> bool:
        infos = self.locals['infos']
        # self.training_env is a VecEnv
        for i, info in enumerate(infos):
            needs_update = False
            
            # Check completion
            if info.get('subgoal_completed', False):
                if self.verbose > 0:
                    print(f"Env {i}: Subgoal completed!")
                self.subgoal_successes += 1
                needs_update = True
            
            # Access wrapper
            env_wrapper = self.training_env.envs[i]

            # Find the wrapper that handles subgoals (has current_subgoal_id)
            # We search top-down to find the outermost wrapper that supports it (e.g. NoiseWrapper delegating)
            target_wrapper = None
            temp_wrapper = env_wrapper

            while True:
                if hasattr(temp_wrapper, 'current_subgoal_id'):
                    target_wrapper = temp_wrapper
                    break
                if hasattr(temp_wrapper, 'env'):
                    temp_wrapper = temp_wrapper.env
                else:
                    break
            
            if target_wrapper is not None:
                if target_wrapper.current_subgoal_id == 0: # NO_OP
                    needs_update = True
                
                if needs_update:
                    # Use the wrapper's get_text_description (supports noise if wrapped)
                    state_text = target_wrapper.get_text_description()
                    subgoal_text = self.planner.generate_subgoal(state_text)
                    subgoal_tuple, subgoal_id = parse_subgoal(subgoal_text)
                    
                    if self.verbose > 0:
                        print(f"Env {i}: New Subgoal: {subgoal_text} -> {subgoal_tuple}")
                    
                    target_wrapper.set_subgoal(subgoal_tuple, subgoal_id)
                    self.total_subgoals += 1
                    
                    # Track Reuse
                    if subgoal_tuple in self.previously_seen_subgoals:
                        self.reused_generation_count += 1
                    
                    self.current_task_subgoals.add(subgoal_tuple)
            
        return True