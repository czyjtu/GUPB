import os
import json
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from reinforced_lib import RLib
from reinforced_lib.exts import BaseExt, observation, parameter
from reinforced_lib.agents.mab import UCB, EGreedy, Softmax
from tqdm import tqdm
from chex import PRNGKey, Array
from typing import Dict, List

from gupb.controller import r2d2
from gupb.__main__ import load_initial_config, configure_logging
from gupb import runner
from gupb.model.weapons import *


MAB_CONFIG_PATH = 'gupb/mab_config.py'
RESULTS_PATH = 'results'
ARENAS_PATH = 'resources/arenas'
SAVE_PATH = 'mab-results.json'

BASE_WEAPON_RANKING = {
    'potion': 0, # 'potion' is not a weapon, but it is the most important item
    'axe': 2,
    'sword': 3,
    'amulet': 4,
    'knife': 5,
    'bow': 6,
    'bow_loaded': 6,
    'bow_unloaded': 6
}

ACTION_MAPPING = {
    0: Axe(),
    1: Sword(),
    2: Amulet(),
    3: Knife(),
    4: Bow()
}

# Configure logging
configure_logging(RESULTS_PATH)
base_config = load_initial_config(MAB_CONFIG_PATH)
enymies_controllers = base_config["controllers"]

class GUPBExt(BaseExt):
    """
    Reinforced-lib extension for tunnig R2D2s parameters in GUPB.
    This extension can be used with Multi-Armed Bandit (MAB) algorithms.
    """

    def __init__(self, n_arms: int) -> None:
        super().__init__()
        self.n = n_arms

    observation_space = gym.spaces.Dict({})

    @parameter(parameter_type=gym.spaces.Box(1, np.inf, (1,), np.int32))
    def n_arms(self) -> int:
        return self.n

    @observation(observation_type=gym.spaces.Box(-np.inf, np.inf, (1,)))
    def reward(self, reward, *args, **kwargs) -> float:
        return reward

def sample_arenas(key: PRNGKey, n_arenas: int = 1) -> List[str]:

    arenas = os.listdir(ARENAS_PATH)
    arenas = list(filter(lambda a: a.startswith('generated_'), arenas))
    sample_idices = jax.random.choice(key, jnp.arange(len(arenas)), (min(n_arenas, len(arenas)),), replace=False)
    
    return [arenas[i].split('.')[0] for i in sample_idices]

def sample_arena(key: PRNGKey, arenas: List[str]) -> str:
    sample_idices = jax.random.choice(key, jnp.arange(len(arenas)), (1,))
    return [arenas[sample_idices[0]]]

def r2d2_score(action: int, base_config: Dict, arena: List[str], verbose: bool = False):

    # Redefine config with new R2D2 decay
    config = base_config.copy()
    config["controllers"] = [r2d2.RecklessRoamingDancingDruid("R2D2", BASE_WEAPON_RANKING)] + enymies_controllers
    config["arenas"] = arena

    # Run the game
    game_runner = runner.Runner(config)
    game_runner.run(ACTION_MAPPING[action])

    if verbose:
        game_runner.print_scores()

    # Return the score as reward
    r2d2_reward = game_runner.scores["RecklessRoamingDancingDruid_R2D2"]
    return r2d2_reward

def run_agent(key: PRNGKey, agent: RLib, n_steps: int, arenas: List[str], agent_name: str):

    key, key_arena = jax.random.split(key)
    
    # Perform first action and get first reward
    act_acc = [0]
    reward_acc = [r2d2_score(act_acc[0], base_config, sample_arena(key_arena, arenas))]

    # Sample actions and rewards for n_steps
    for _ in tqdm(range(n_steps), desc=f"{agent_name}"):
        key, key_arena = jax.random.split(key)

        # Sample action from mab agent
        act = agent.sample(reward_acc[-1]).item()

        # Get reward for action
        reward = r2d2_score(
            act, base_config, 
            sample_arena(key_arena, arenas
        ))

        # Accumulate action-reward pair
        act_acc.append(int(act))
        reward_acc.append(reward)
    
    return act_acc, reward_acc

def main():

    agents = {
        # 'EGreedy': RLib(
        #     agent_type=EGreedy,
        #     agent_params={'e': 0.166, 'optimistic_start': 2000.},
        #     ext_type=GUPBExt,
        #     ext_params={'n_arms': 5}
        # ),
        # 'UCB': RLib(
        #     agent_type=UCB,
        #     agent_params={'c': 100},
        #     ext_type=GUPBExt,
        #     ext_params={'n_arms': 5}
        # ),
        'Softmax': RLib(
            agent_type=Softmax,
            agent_params={'lr': 1., 'tau': 25., 'multiplier': 0.01},
            ext_type=GUPBExt,
            ext_params={'n_arms': 5}
        )
    }

    # Prepare arenas and results accumulator
    key = jax.random.PRNGKey(2137)
    key, key_arenas = jax.random.split(key)
    arenas = sample_arenas(key_arenas, n_arenas=50)
    results_dict = {}

    # Train each agent for n_steps
    n_steps = 1000
    for agent_name, agent in agents.items():
        act_acc, reward_acc = run_agent(key, agent, n_steps, arenas, agent_name)
        results_dict[agent_name] = {
            "actions": act_acc,
            "rewards": reward_acc
        }

        # Checkpoint results
        with open(f"mab-results-softmax.json", 'w') as fp:
            json.dump(results_dict, fp)


if __name__ == "__main__":
    main()