import gymnasium as gym
import numpy as np
import random

class ObservationNoiseWrapper(gym.ObservationWrapper):
    """
    Adds noise to observations (Image and Text).

    Args:
        env (gym.Env): The environment to wrap.
        noise_type (str): 'gaussian' (image), 'text_mask' (text), 'text_replace' (text).
        intensity (float): Standard deviation for Gaussian or probability for text masks.
    """
    def __init__(self, env, noise_type='gaussian', intensity=0.0):
        super().__init__(env)
        self.noise_type = noise_type
        self.intensity = intensity

    def observation(self, obs):
        if self.intensity <= 0:
            return obs

        new_obs = obs.copy()

        # Image Noise
        if self.noise_type == 'gaussian' and 'image' in new_obs:
            image = new_obs['image'].astype(np.float32)
            noise = np.random.normal(0, self.intensity * 255, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
            new_obs['image'] = image

        return new_obs

    def get_text_description(self):
        """
        Intercepts the text description to add noise.
        """
        if not hasattr(self.env, 'get_text_description'):
            return ""

        desc = self.env.get_text_description()

        if self.intensity <= 0:
            return desc

        if self.noise_type == 'text_mask':
            # Drop words with probability 'intensity'
            words = desc.split()
            new_words = [w if random.random() > self.intensity else "[MASK]" for w in words]
            return " ".join(new_words)

        elif self.noise_type == 'text_replace':
             # Replace characters or words
             words = desc.split()
             new_words = []
             for w in words:
                 if random.random() < self.intensity:
                     # Replace with random string
                     new_words.append("".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=len(w))))
                 else:
                     new_words.append(w)
             return " ".join(new_words)

        return desc

    def __getattr__(self, name):
        return getattr(self.env, name)

class TransitionNoiseWrapper(gym.ActionWrapper):
    """
    Adds stochasticity to transitions (action slip).

    Args:
        env (gym.Env): The environment to wrap.
        slip_prob (float): Probability of executing a random action instead of the intended one.
    """
    def __init__(self, env, slip_prob=0.0):
        super().__init__(env)
        self.slip_prob = slip_prob

    def action(self, action):
        if self.slip_prob > 0 and random.random() < self.slip_prob:
            # Sample random action from space
            return self.action_space.sample()
        return action

    def __getattr__(self, name):
        return getattr(self.env, name)

class RewardNoiseWrapper(gym.RewardWrapper):
    """
    Adds noise to the reward signal.

    Args:
        env (gym.Env): The environment to wrap.
        noise_std (float): Standard deviation of Gaussian noise added to reward.
    """
    def __init__(self, env, noise_std=0.0):
        super().__init__(env)
        self.noise_std = noise_std

    def reward(self, reward):
        if self.noise_std > 0:
            return reward + np.random.normal(0, self.noise_std)
        return reward

    def __getattr__(self, name):
        return getattr(self.env, name)
