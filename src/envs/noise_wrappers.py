import gymnasium as gym
import numpy as np

class ObservationNoiseWrapper(gym.ObservationWrapper):
    def __init__(self, env, text_noise_level=0.0, text_noise_type='mask', image_noise_level=0.0):
        super().__init__(env)
        self.text_noise_level = text_noise_level
        self.text_noise_type = text_noise_type
        self.image_noise_level = image_noise_level

    def observation(self, obs):
        # Modify image in obs
        if self.image_noise_level > 0:
            if isinstance(obs, dict) and 'image' in obs:
                obs['image'] = self._add_gaussian_noise(obs['image'])
            elif isinstance(obs, np.ndarray):
                obs = self._add_gaussian_noise(obs)
        return obs

    def _add_gaussian_noise(self, image):
        noise = np.random.normal(0, self.image_noise_level, image.shape).astype(np.float32)
        noisy_image = image.astype(np.float32) + noise
        return np.clip(noisy_image, 0, 255).astype(image.dtype)

    def get_text_description(self):
        # Look for get_text_description in env or unwrapped
        if hasattr(self.env, 'get_text_description'):
             desc = self.env.get_text_description()
        elif hasattr(self.unwrapped, 'get_text_description'):
             desc = self.unwrapped.get_text_description()
        else:
             return ""

        if self.text_noise_level > 0:
            return self._apply_text_noise(desc)
        return desc

    def _apply_text_noise(self, text):
        words = text.split()
        if not words: return text

        num_mask = int(len(words) * self.text_noise_level)
        if num_mask == 0 and self.text_noise_level > 0 and len(words) > 0: num_mask = 1

        indices = np.random.choice(len(words), num_mask, replace=False)

        if self.text_noise_type == 'mask':
            for i in indices:
                words[i] = "[MASK]"
        elif self.text_noise_type == 'replace':
             noise_tokens = ["blue", "red", "key", "door", "ball", "wall", "floor", "nothing"]
             for i in indices:
                 words[i] = np.random.choice(noise_tokens)

        return " ".join(words)

    def __getattr__(self, name):
        return getattr(self.env, name)

class TransitionNoiseWrapper(gym.Wrapper):
    def __init__(self, env, transition_noise_level=0.0):
        super().__init__(env)
        self.transition_noise_level = transition_noise_level

    def step(self, action):
        if self.transition_noise_level > 0 and np.random.rand() < self.transition_noise_level:
            # Action randomization
            if hasattr(self.action_space, 'n'):
                action = np.random.randint(self.action_space.n)
            elif hasattr(self.action_space, 'sample'):
                 action = self.action_space.sample()

        return self.env.step(action)

    def __getattr__(self, name):
        return getattr(self.env, name)

class RewardNoiseWrapper(gym.Wrapper):
    def __init__(self, env, reward_noise_level=0.0):
        super().__init__(env)
        self.reward_noise_level = reward_noise_level

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.reward_noise_level > 0:
             noise = np.random.normal(0, self.reward_noise_level)
             reward += noise

        return obs, reward, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)
