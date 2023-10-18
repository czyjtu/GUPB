import numpy as np
import gym
import gym.spaces
from gym import ObservationWrapper


from gupb.gym_env.gym_env import GUPBEnv
from gupb.model.arenas import Arena
from gupb.model.characters import ChampionKnowledge


LARGEST_ARENA_SHAPE = (100, 100)

tiles_mapping = {
    "out": 0,
    "land": 1,
    "sea": 2,
    "wall": 3,
    "menhir": 4,
    "champion": 5,
    "knife": 6,
    "sword": 7,
    "bow": 8,
    "axe": 9,
    "amulet": 10,
    "potion": 11,
    "enymy": 12,
    "mist": 13,
}

class GUPBEnvMatrix(ObservationWrapper):

    def _init_matrix(self, arena: Arena):
        self.arena_shape = arena.size[1], arena.size[0]
        for coords, tile in arena.terrain.items():
            self.matrix[coords.y, coords.x] = tiles_mapping[tile.description().type]
        self.initial_arena = self.matrix.copy()

    def _fill_matrix(self, champion_knowledge: ChampionKnowledge):
        
        # Update Visible tiles
        for coords, tile_description in champion_knowledge.visible_tiles.items():
            
            if tile_description.loot:
                self.matrix[coords.y, coords.x] = tiles_mapping[tile_description.loot.name]
            
            if tile_description.consumable:
                self.matrix[coords.y, coords.x] = tiles_mapping[tile_description.consumable.name]
            
            if tile_description.character:
                self.matrix[coords.y, coords.x] = tiles_mapping["enymy"]
            
            if tile_description.effects:
                if "mist" in tile_description.effects:
                    self.matrix[coords.y, coords.x] = tiles_mapping["mist"]

        # Update Champion position
        self.matrix[champion_knowledge.position.y, champion_knowledge.position.x] = tiles_mapping["champion"]

    def _decay_step(self, champion_knowledge: ChampionKnowledge):
        
        # Decay the whole mask
        self.decay_mask = np.maximum(self.decay_mask - 1, 0)

        # Reset decayed tiles
        self.matrix = np.where(self.decay_mask == 0, self.initial_arena, self.matrix)

        # Reset decay of visible tiles
        for coords, tile_description in champion_knowledge.visible_tiles.items():
            self.decay_mask[coords.y, coords.x] = self.decay


    def __init__(self, env: GUPBEnv, decay=0):
        super().__init__(env)

        # The biggest map up-to-date is 100x100, hence the shape
        # TODO: Is it enough? Can we get agent health and facing direction?
        self.observation_space = gym.spaces.Box(
            low=0, high=len(tiles_mapping.keys()), shape=LARGEST_ARENA_SHAPE, dtype=np.int8
        )

        self.arena_id = None
        self.arena_shape = None
        self.initial_arena = None

        # Controls the memory of the agent. Decay of n, means that the agent will assume that
        # the item is still there for n steps, even when it's not visible. Note that the decay
        # of 1 means no memory at all, the higher the decay, the longer the memory.
        self.decay = decay
        self.decay_mask = np.zeros(LARGEST_ARENA_SHAPE, np.int8)

        # This is the representation of the map state that will be returned as the observation
        self.matrix = np.zeros(LARGEST_ARENA_SHAPE, dtype=np.int8)

    def observation(self, observation):
        
        # Initialize the matrix with the arena when it's first seen
        if self.arena_id is None:
            self.arena_id = observation["arena"]
            self._init_matrix(Arena.load(self.arena_id))
        
        # Apply decay and update the decay mask
        self._decay_step(observation["view"])
        
        # Update the matrix with the current observation
        self._fill_matrix(observation["view"])

        return self.matrix

            