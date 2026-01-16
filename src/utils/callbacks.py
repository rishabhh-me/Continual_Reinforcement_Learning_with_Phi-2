from stable_baselines3.common.callbacks import BaseCallback
from src.utils.subgoal_parser import parse_subgoal
from src.envs.wrappers import SubgoalWrapper

class SubgoalUpdateCallback(BaseCallback):
    def __init__(self, planner, verbose=0):
        super(SubgoalUpdateCallback, self).__init__(verbose)
        self.planner = planner
        self.subgoal_successes = 0
        self.total_subgoals = 0

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
            # Unwrap if needed
            while hasattr(env_wrapper, 'env'):
                if isinstance(env_wrapper, SubgoalWrapper):
                    break
                env_wrapper = env_wrapper.env

            if isinstance(env_wrapper, SubgoalWrapper):
                if env_wrapper.current_subgoal_id == 0: # NO_OP
                    needs_update = True

                if needs_update:
                    state_text = env_wrapper.get_text_description()
                    subgoal_text = self.planner.generate_subgoal(state_text)
                    subgoal_tuple, subgoal_id = parse_subgoal(subgoal_text)

                    if self.verbose > 0:
                        print(f"Env {i}: New Subgoal: {subgoal_text} -> {subgoal_tuple}")

                    env_wrapper.set_subgoal(subgoal_tuple, subgoal_id)
                    self.total_subgoals += 1

        return True
