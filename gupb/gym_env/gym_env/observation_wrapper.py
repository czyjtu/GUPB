import numpy as np
import gymnasium as gym
import gymnasium.spaces
from gymnasium import ObservationWrapper, spaces
gymnasium.wrappers


from gupb.gym_env.gym_env import GUPBEnv
from gupb.model.arenas import Arena
from gupb.model.characters import ChampionKnowledge


LARGEST_ARENA_SHAPE = (100, 100)

MAX_TILE_ID = 50

tiles_mapping = {
    "mist": 0,

    "out": 3,
    "land": 4,
    "sea": 5,
    "wall": 6,

    "knife": 10,
    "sword": 11,
    "bow_unloaded": 12,
    "bow_loaded": 13, # "bow_unloaded" and "bow_loaded" are the same tile
    "axe": 14,
    "amulet": 15,

    "enymy": 17,
    "potion": 19,

    "champion": 25,

    "menhir": MAX_TILE_ID,# don't change this value
}

class ImageWrapper(ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=env.observation_space.shape, dtype=np.float32
        )

    def observation(self, observation: np.ndarray):
        return (observation / MAX_TILE_ID).astype(np.float32)

class GUPBEnvMatrix(ObservationWrapper):

    def _init_matrix(self, arena: Arena):
        self.arena_shape = arena.size[1], arena.size[0]
        for coords, tile in arena.terrain.items():
            self.matrix[coords[1], coords[0]] = tiles_mapping[tile.description().type]
        self.initial_arena = self.matrix.copy()

    def _fill_matrix(self, champion_knowledge: ChampionKnowledge):
        
        # Update Visible tiles
        for coords, tile_description in champion_knowledge.visible_tiles.items():

            if tile_description.type == "menhir":
                self.matrix[coords[1], coords[0]] = tiles_mapping[tile_description.type]
                self.menhir_position = (coords[1], coords[0])
            
            if tile_description.loot:
                self.matrix[coords[1], coords[0]] = tiles_mapping[tile_description.loot.name]
            
            if tile_description.consumable:
                self.matrix[coords[1], coords[0]] = tiles_mapping[tile_description.consumable.name]
            
            if tile_description.character:
                self.matrix[coords[1], coords[0]] = tiles_mapping["enymy"]
            
            if tile_description.effects:
                if "mist" in tile_description.effects:
                    self.matrix[coords[1], coords[0]] = tiles_mapping["mist"]

        # Update Champion position
        self.matrix[champion_knowledge.position.y, champion_knowledge.position.x] = tiles_mapping["champion"]

    def _decay_step(self, champion_knowledge: ChampionKnowledge):
        
        # Decay the whole mask
        self.decay_mask = np.maximum(self.decay_mask - 1, 0)

        # Reset decayed tiles
        self.matrix = np.where(self.decay_mask == 0, self.initial_arena, self.matrix)
        # - but keep the menhir in place once discovered
        if self.menhir_position:
            self.matrix[self.menhir_position[0], self.menhir_position[1]] = tiles_mapping["menhir"]

        # Reset decay of visible tiles
        for coords, tile_description in champion_knowledge.visible_tiles.items():
            self.decay_mask[coords[1], coords[0]] = self.decay


    def __init__(self, env: GUPBEnv, decay=0):
        super().__init__(env)

        # The biggest map up-to-date is 100x100, hence the shape
        # TODO: Is it enough? Can we get agent health and facing direction?
        self.observation_space = gym.spaces.Box(
            low=0, high=max(tiles_mapping.values()), shape=LARGEST_ARENA_SHAPE, dtype=np.int8
        )

        self.arena_id = None
        self.arena_shape = None
        self.initial_arena = None
        self.menhir_position = None

        # Controls the memory of the agent. Decay of n, means that the agent will assume that
        # the item is still there for n steps, even when it's not visible. Note that the decay
        # of 1 means no memory at all, the higher the decay, the longer the memory.
        # TODO Could we use a different decay for dynamic characters and a different one for static items?
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

            