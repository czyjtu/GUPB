import numpy as np
import pytest 
import gym 

from gupb.controller import random
import gupb.gym_env.gym_env
from gupb.model.characters import ChampionKnowledge


def test_env_build_successfully():
    env: gym.Env = gym.make('GUPB-v0')

    assert env is not None
    assert env.reset() is not None


def test_reset_return_arena_and_view():
    env: gym.Env = gym.make('GUPB-v0')

    obs = env.reset()

    assert obs is not None
    match obs:
        case {"arena": _, "view": _}: 
            pass
        case _:
            assert False, obs

def test_observation_contains_actual_view():
    env: gym.Env = gym.make('GUPB-v0')

    obs = env.reset()

    assert obs is not None
    assert isinstance(obs["view"], ChampionKnowledge)
