import unittest
import numpy as np
import gymnasium as gym
from src.envs.noise_wrappers import ObservationNoiseWrapper, TransitionNoiseWrapper, RewardNoiseWrapper
from src.envs.wrappers import SubgoalWrapper

class MockEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(low=0, high=255, shape=(10, 10, 3), dtype=np.uint8)
        })
        self.action_space = gym.spaces.Discrete(5)
    def reset(self, **kwargs):
        return self.observation_space.sample(), {}
    def step(self, action):
        return self.observation_space.sample(), 1.0, False, False, {}
    def get_text_description(self):
        return "This is a test description."
    def set_subgoal(self, t, i):
        self.subgoal = (t, i)

class TestPhase6Wrappers(unittest.TestCase):
    def test_observation_noise(self):
        env = MockEnv()
        env = ObservationNoiseWrapper(env, noise_type='gaussian', intensity=0.1)
        obs, _ = env.reset()
        self.assertEqual(obs['image'].shape, (10, 10, 3))

        # Text Noise
        env = MockEnv()
        env = ObservationNoiseWrapper(env, noise_type='text_mask', intensity=1.0) # Always mask
        desc = env.get_text_description()
        self.assertIn("[MASK]", desc)

    def test_transition_noise(self):
        env = MockEnv()
        env = TransitionNoiseWrapper(env, slip_prob=0.5)
        # Just check it runs
        env.step(0)

    def test_reward_noise(self):
        env = MockEnv()
        env = RewardNoiseWrapper(env, noise_std=0.1)
        _, r, _, _, _ = env.step(0)
        self.assertNotEqual(r, 1.0) # Probabilistic, but highly likely not 1.0

    def test_delegation(self):
        env = MockEnv()
        env = ObservationNoiseWrapper(env)
        env.set_subgoal(("test",), 1)
        self.assertEqual(env.subgoal, (("test",), 1))

if __name__ == '__main__':
    unittest.main()
