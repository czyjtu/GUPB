import gym 
from gym.envs.registration import register

from gupb import controller


class GUPBEnv(gym.Env):
    def __init__(
        self, 
        config: dict[str, int] | None = None,
        controllers: list[controller.Controller] | None = None,
    ):
        self.config = config
        self.controllers = controllers

    @property
    def action_space(self):
        return gym.spaces.Discrete(3)
    
    @property
    def observation_space(self):
        return gym.spaces.Discrete(3)

    def reset(self):
        return 1

    def step(self, action):
        pass

    def render(self, mode='human'):
        pass

    def close(self):
        pass 


register(
     id='GUPB-v0',
     entry_point='gupb.gym_env.gym_env:GUPBEnv',
)
