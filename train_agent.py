from gupb.gym_env.gym_env.observation_wrapper import GUPBEnvMatrix, ImageWrapper
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
from gymnasium.wrappers import frame_stack


RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
CHECKPOINT_DIR = Path("checkpoints") / RUN_ID
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)


@dataclasses.dataclass
class TrainingConfig:
    last_n_frames: int = 3
    lr: float = 1e-3
    max_episodes: int = 1_000 
    features_dim: int = 64
    extractor_arch: list[int] = dataclasses.field(default_factory=lambda: [16, 32, 64])
    actor_arch: list[int] = dataclasses.field(default_factory=lambda: [64, 64])
    critic_arch: list[int] = dataclasses.field(default_factory=lambda: [64, 64])

def wrap_env(env, config: TrainingConfig):
    env = GUPBEnvMatrix(env, 0)
    env = ImageWrapper(env)
    env = frame_stack.FrameStack(env, config.last_n_frames)
    print(env.observation_space.shape)
    return env 


def main(config: TrainingConfig, run):
    env = gym.make(
        "GUPB-v0",
        config=EnvConfig(
            arenas=["lone_sanctum", "ordinary_chaos"], 
            controllers=[random.RandomController(f"Alice{i}") for i in range(10)]
        )
    )
    env = wrap_env(env, config)
    model = PPO(
        CustomActorCriticPolicy,
        env,
        policy_kwargs=dict(
            normalize_images=False,
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(
                features_dim=config.features_dim, 
                architecture=config.extractor_arch
            ),
            actor_arch=config.actor_arch,
            critic_arch=config.critic_arch,
        ),
        verbose=1,
        learning_rate=config.lr,
        tensorboard_log=f"runs/{run.id}",
    )
    model.learn(
        total_timesteps=config.max_episodes * 1000, 
        callback=WandbCallback(
            gradient_save_freq=1000,
            model_save_path=f"models/{run.id}",
            log="all",
            verbose=1,
            model_save_freq=1000,
    ),
)
    torchPolicy = PPO2TorchPolicy(model).eval()
    th.save(torchPolicy, CHECKPOINT_DIR / "test_torch_policy.pth")

if __name__ == "__main__":
    run = wandb.init(
        project="gupb", 
        entity="czyjtu", 
        name=RUN_ID,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,  
        )
    config = TrainingConfig(
        max_episodes=1_000,
        features_dim=128,
        extractor_arch=[16, 32, 64],
        actor_arch=[64, 64],
        critic_arch=[64, 64],
    )
    main(config, run)
    