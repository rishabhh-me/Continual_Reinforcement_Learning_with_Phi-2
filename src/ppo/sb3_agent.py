import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_device
import torch
import torch.nn as nn
import sys

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor for MiniGrid + Subgoal.
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        obs_shape = observation_space["image"].shape
        # Check if channels first (C, H, W) or channels last (H, W, C)
        # MiniGrid C=3.
        if obs_shape[0] in [1, 3]: # Channels First
             n_input_channels = obs_shape[0]
             self.is_channels_first = True
        else: # Channels Last
             n_input_channels = obs_shape[2]
             self.is_channels_first = False

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample = torch.as_tensor(observation_space["image"].sample()[None]).float()
            if not self.is_channels_first:
                sample = sample.permute(0, 3, 1, 2)
            n_flatten = self.cnn(sample).shape[1]

        self.subgoal_embedding = nn.Embedding(31, 16) 

        self.linear = nn.Sequential(
            nn.Linear(n_flatten + 16, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        image = observations["image"].float()
        if not self.is_channels_first:
            image = image.permute(0, 3, 1, 2)

        img_feats = self.cnn(image)

        subgoal = observations["subgoal"].long().squeeze(1) 
        sub_feats = self.subgoal_embedding(subgoal)

        combined = torch.cat([img_feats, sub_feats], dim=1)

        return self.linear(combined)

class FlatMinigridFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom Feature Extractor for MiniGrid (No Subgoal).
    """
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        obs_shape = observation_space["image"].shape
        if obs_shape[0] in [1, 3]: # Channels First
             n_input_channels = obs_shape[0]
             self.is_channels_first = True
        else: # Channels Last
             n_input_channels = obs_shape[2]
             self.is_channels_first = False

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.as_tensor(observation_space["image"].sample()[None]).float()
            if not self.is_channels_first:
                sample = sample.permute(0, 3, 1, 2)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        image = observations["image"].float()
        if not self.is_channels_first:
            image = image.permute(0, 3, 1, 2)
        return self.linear(self.cnn(image))

def create_agent(env, config, agent_type="ppo"):
    """
    Creates and returns a PPO agent configured for the environment.
    """
    
    if agent_type == "flat":
        policy_kwargs = dict(
            features_extractor_class=FlatMinigridFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        )
        AgentClass = PPO
    else:
        policy_kwargs = dict(
            features_extractor_class=MinigridFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        )
        if agent_type == "rehearsal":
            from src.ppo.rehearsal_agent import RehearsalPPO
            AgentClass = RehearsalPPO
        else:
            AgentClass = PPO

    agent = AgentClass(
        "MultiInputPolicy",
        env,
        learning_rate=config['rl']['learning_rate'],
        n_steps=config['rl']['n_steps'],
        batch_size=config['rl']['batch_size'],
        n_epochs=config['rl']['n_epochs'],
        gamma=config['rl']['gamma'],
        gae_lambda=config['rl']['gae_lambda'],
        clip_range=config['rl']['clip_range'],
        ent_coef=config['rl']['ent_coef'],
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=config['experiment']['device']
    )

    return agent