import queue
import threading
import time
import numpy as np
import pytest 
import gym 

from gupb.controller import random
from  gupb.gym_env.gym_env import AgentAction, EnvConfig, GUPBEnv, QueueController
from gupb.gym_env.gym_env.observation_wrapper import GUPBEnvMatrix, LARGEST_ARENA_SHAPE
from gupb.model import characters
from gupb.model.characters import ChampionKnowledge


@pytest.fixture
def wrapped_env():
    base_env = gym.make(
        "GUPB-v0", 
        config=EnvConfig(
            arenas=["lone_sanctum"], 
            controllers=[random.RandomController("Alice"), random.RandomController("Bob")]
        )
    )
    env = GUPBEnvMatrix(base_env, decay=3)
    yield env
    env.close()


@pytest.fixture
def wrapped_env_instant_decay():
    base_env = gym.make(
        "GUPB-v0", 
        config=EnvConfig(
            arenas=["lone_sanctum"], 
            controllers=[random.RandomController("Alice"), random.RandomController("Bob")]
        )
    )
    env = GUPBEnvMatrix(base_env, decay=0)
    yield env
    env.close()


def test_observation_reset(wrapped_env):
    obs, _ = wrapped_env.reset()
    assert obs.shape == LARGEST_ARENA_SHAPE


def test_observation_step(wrapped_env):
    wrapped_env.reset()

    # Step 1 time
    obs, *_ = wrapped_env.step(AgentAction.DO_NOTHING)
    assert obs.shape == LARGEST_ARENA_SHAPE

    # Step 3 more times
    wrapped_env.step(AgentAction.DO_NOTHING)
    wrapped_env.step(AgentAction.DO_NOTHING)
    obs, *_ = wrapped_env.step(AgentAction.DO_NOTHING)
    assert obs.shape == LARGEST_ARENA_SHAPE


def test_decay(wrapped_env):
    i = 0
    obs, _ = wrapped_env.reset()
    np.savetxt(f"observation_{i}.txt", obs[:19,:19], fmt="%02d")

    while i < 10:
        obs, *_ = wrapped_env.step(AgentAction.TURN_LEFT)
        np.savetxt(f"observation_{i}.txt", obs[:19,:19], fmt="%02d")
        i += 1


def test_instant_decay(wrapped_env_instant_decay):
    i = 0
    obs, _ = wrapped_env_instant_decay.reset()
    np.savetxt(f"observation_instant_decay_{i}.txt", obs[:19,:19], fmt="%02d")

    while i < 10:
        obs, *_ = wrapped_env_instant_decay.step(AgentAction.TURN_LEFT)
        np.savetxt(f"observation_instant_decay_{i}.txt", obs[:19,:19], fmt="%02d")
        i += 1

