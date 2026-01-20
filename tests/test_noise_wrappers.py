import unittest
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.envs.noise_wrappers import ObservationNoiseWrapper

class MockEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (10, 10, 3), dtype=np.uint8)
        })
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        return {"image": np.zeros((10,10,3), dtype=np.uint8)}, {}

    def step(self, action):
        return self.reset()[0], 0.0, False, False, {}

    def get_text_description(self):
        return "this is a test description"

class TestNoiseWrappers(unittest.TestCase):
    def test_text_mask_noise(self):
        env = MockEnv()
        wrapper = ObservationNoiseWrapper(env, text_noise_level=0.5, text_noise_type='mask')

        # Test multiple times
        changed = False
        for _ in range(10):
            desc = wrapper.get_text_description()
            if "[MASK]" in desc:
                changed = True
                break
        self.assertTrue(changed, "Text masking failed to produce [MASK]")

    def test_text_replace_noise(self):
        env = MockEnv()
        wrapper = ObservationNoiseWrapper(env, text_noise_level=1.0, text_noise_type='replace')
        desc = wrapper.get_text_description()
        self.assertNotEqual(desc, "this is a test description")
        # Ensure it contains some noise token like "blue", "red", etc.
        noise_tokens = ["blue", "red", "key", "door", "ball", "wall", "floor", "nothing"]
        self.assertTrue(any(token in desc for token in noise_tokens))

    def test_image_noise(self):
        env = MockEnv()
        wrapper = ObservationNoiseWrapper(env, image_noise_level=50.0) # High noise
        obs, _ = wrapper.reset()
        # Should not be all zeros (Gaussian noise added)
        # 0 + noise, clipped.

        # Check if any value > 0
        self.assertFalse(np.all(obs['image'] == 0), f"Image is all zeros! Mean: {np.mean(obs['image'])}")

if __name__ == '__main__':
    unittest.main()
