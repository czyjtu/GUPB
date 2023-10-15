import pytest 
import gym 

import gupb.gym_env.gym_env

@pytest.fixture
def env() -> gym.Env:
    return gym.make('GUPB-v0')

def test_env_build_successfully(env: gym.Env):
    assert env is not None
    assert env.reset() is not None
