import gym 
from gym.envs.registration import register
import gym.spaces 

from gupb import controller
from gupb.model.characters import ChampionKnowledge
from gupb.model.coordinates import Coords

class ViewSpace(gym.spaces.Space):
    # this class is just a boilerplate to make gym happy
    def __init__(self):
        super().__init__()

    def sample(self):
        return None

    def contains(self, x):
        return isinstance(x, ChampionKnowledge)

    def __repr__(self):
        return f"ViewSpace"

    def __eq__(self, other):
        return isinstance(other, ViewSpace)


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
        return gym.spaces.Dict({"arena": gym.spaces.Discrete(1), "view": ViewSpace()})

    def reset(self):
        return {"arena": 0, "view": ChampionKnowledge(Coords(0, 0), 0, {})}

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
