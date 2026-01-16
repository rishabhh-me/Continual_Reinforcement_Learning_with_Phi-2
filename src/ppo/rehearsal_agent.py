import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from copy import deepcopy

class RehearsalPPO(PPO):
    """
    PPO variant that supports rehearsal (experience replay) from previous tasks.
    It maintains a storage of data from previous tasks and mixes it into the training batch.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_storage = []
        self.rehearsal_fraction = 0.3

    def cache_current_task_data(self):
        """
        Collects data from current policy and stores it for rehearsal.
        Should be called at the end of a task.
        """
        print("Caching data for rehearsal...")
        self.rollout_buffer.reset()

        # Disable gradient calculation for collection if needed (SB3 does it internally)
        self.collect_rollouts(self.env, callback=None, rollout_buffer=self.rollout_buffer, n_rollout_steps=self.n_steps)

        # Deep copy the buffer data
        cached_data = {
            'observations': deepcopy(self.rollout_buffer.observations),
            'actions': deepcopy(self.rollout_buffer.actions),
            'old_log_prob': deepcopy(self.rollout_buffer.log_probs),
            'advantages': deepcopy(self.rollout_buffer.advantages),
            'returns': deepcopy(self.rollout_buffer.returns),
            'values': deepcopy(self.rollout_buffer.values)
        }
        self.replay_storage.append(cached_data)
        # Reset buffer again so it's clean for next use (though train loop usually handles fill->train->reset)
        self.rollout_buffer.reset()

    def train(self):
        if len(self.replay_storage) > 0:
            n_replay = int(self.n_steps * self.rehearsal_fraction)

            # Create lists for concatenation
            current_obs = self.rollout_buffer.observations
            is_dict = isinstance(current_obs, dict)

            if is_dict:
                mixed_obs = {k: [v] for k, v in current_obs.items()}
            else:
                mixed_obs = [current_obs]

            mixed_actions = [self.rollout_buffer.actions]
            mixed_log_probs = [self.rollout_buffer.log_probs]
            mixed_advantages = [self.rollout_buffer.advantages]
            mixed_returns = [self.rollout_buffer.returns]
            mixed_values = [self.rollout_buffer.values]

            # Sample from storage
            # Distribute n_replay across stored tasks
            samples_per_task = max(1, n_replay // len(self.replay_storage))

            for task_data in self.replay_storage:
                # Task data shapes are (n_steps, n_envs, ...)
                available_steps = task_data['actions'].shape[0]
                # Randomly sample indices
                t_indices = np.random.choice(available_steps, min(available_steps, samples_per_task), replace=True)

                if is_dict:
                     for k in mixed_obs:
                         mixed_obs[k].append(task_data['observations'][k][t_indices])
                else:
                    mixed_obs.append(task_data['observations'][t_indices])

                mixed_actions.append(task_data['actions'][t_indices])
                mixed_log_probs.append(task_data['old_log_prob'][t_indices])
                mixed_advantages.append(task_data['advantages'][t_indices])
                mixed_returns.append(task_data['returns'][t_indices])
                mixed_values.append(task_data['values'][t_indices])

            # Concatenate
            if is_dict:
                final_obs = {k: np.concatenate(v, axis=0) for k, v in mixed_obs.items()}
            else:
                final_obs = np.concatenate(mixed_obs, axis=0)

            final_actions = np.concatenate(mixed_actions, axis=0)
            final_log_probs = np.concatenate(mixed_log_probs, axis=0)
            final_advantages = np.concatenate(mixed_advantages, axis=0)
            final_returns = np.concatenate(mixed_returns, axis=0)
            final_values = np.concatenate(mixed_values, axis=0)

            # Save original buffer state
            orig_obs = self.rollout_buffer.observations
            orig_actions = self.rollout_buffer.actions
            orig_log_probs = self.rollout_buffer.log_probs
            orig_advantages = self.rollout_buffer.advantages
            orig_returns = self.rollout_buffer.returns
            orig_values = self.rollout_buffer.values
            orig_size = self.rollout_buffer.buffer_size

            # Inject mixed data
            self.rollout_buffer.observations = final_obs
            self.rollout_buffer.actions = final_actions
            self.rollout_buffer.log_probs = final_log_probs
            self.rollout_buffer.advantages = final_advantages
            self.rollout_buffer.returns = final_returns
            self.rollout_buffer.values = final_values
            self.rollout_buffer.buffer_size = final_actions.shape[0]

            # Train with extended buffer
            super().train()

            # Restore original buffer
            self.rollout_buffer.observations = orig_obs
            self.rollout_buffer.actions = orig_actions
            self.rollout_buffer.log_probs = orig_log_probs
            self.rollout_buffer.advantages = orig_advantages
            self.rollout_buffer.returns = orig_returns
            self.rollout_buffer.values = orig_values
            self.rollout_buffer.buffer_size = orig_size

        else:
            super().train()
