from wandb.integration.sb3 import WandbCallback
import wandb 
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import PPO 
import gymnasium as gym 
import dataclasses
import torch as th 
from gupb.gym_env.agent.actor_critic import CustomActorCriticPolicy, CustomFeatureExtractor, PPO2TorchPolicy 
from pathlib import Path 
from datetime import datetime 
from  gupb.gym_env.gym_env import AgentAction, EnvConfig, GUPBEnv, QueueController
from gupb.controller import random


RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
CHECKPOINT_DIR = Path("checkpoints") / RUN_ID
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)


@dataclasses.dataclass
class TrainingConfig:
    max_episodes: int = 1_000 
    features_dim: int = 64
    actor_arch: list[int] = dataclasses.field(default_factory=lambda: [64, 64])
    critic_arch: list[int] = dataclasses.field(default_factory=lambda: [64, 64])

def wrap_env(env):
    return env

def main(config: TrainingConfig):
    env = gym.make(
        "GUPB-v0",
        config=EnvConfig(
            arenas=["lone_sanctum", "ordinary_chaos"], 
            controllers=[random.RandomController(f"Alice{i}") for i in range(10)]
        )
    )
    env = wrap_env(env)
    model = PPO(
        CustomActorCriticPolicy,
        env,
        policy_kwargs=dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=config.features_dim),
            actor_arch=config.actor_arch,
            critic_arch=config.critic_arch,
        ),
    )
    torchPolicy = PPO2TorchPolicy(model).eval()
    th.save(torchPolicy, CHECKPOINT_DIR / "test_torch_policy.pth")

if __name__ == "__main__":
    config = TrainingConfig(
        max_episodes=1_000,
        features_dim=64,
        actor_arch=[64, 64],
        critic_arch=[64, 64],
    )
    main(config)
    