from dataclasses import dataclass
from enum import Enum
import random
import gym 
from gym.envs.registration import register
import gym.spaces 
import queue 
import threading


from gupb import controller
from gupb.model import characters
from gupb.model import arenas, games
from gupb.model.characters import ChampionKnowledge
from gupb.gym_env.gym_env.queue_controller import QueueController


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
    
class DescriptionSpace(gym.spaces.Space):
    # this class is just a boilerplate to make gym happy
    def __init__(self):
        super().__init__()

    def sample(self):
        return None

    def contains(self, x):
        return isinstance(x, str)

    def __repr__(self):
        return f"ViewSpace"

    def __eq__(self, other):
        return isinstance(other, DescriptionSpace)
    
    
class AgentAction(Enum):
    TURN_LEFT = 0
    TURN_RIGHT = 1
    STEP_FORWARD = 2
    ATTACK = 3
    DO_NOTHING = 4

AgentAction2Action = {
    AgentAction.TURN_LEFT: characters.Action.TURN_LEFT,
    AgentAction.TURN_RIGHT: characters.Action.TURN_RIGHT,
    AgentAction.STEP_FORWARD: characters.Action.STEP_FORWARD,
    AgentAction.ATTACK: characters.Action.ATTACK,
    AgentAction.DO_NOTHING: characters.Action.DO_NOTHING,
}

@dataclass 
class EnvConfig:
    arenas: list[str]
    controllers: list[controller.Controller]
    start_balancing: bool = True 


def run_game(game, stop_event: threading.Event):
    while not stop_event.is_set() and not game.finished:
        game.cycle()


class GUPBEnv(gym.Env):
    OBSERVATION_TIMEOUT = 5.0

    def __init__(
        self, 
        config: EnvConfig
    ):
        self.config: EnvConfig = config
        self.controllers_order = list(range(len(self.config.controllers) + 1)) # index 0 = queue controller
        self.game_thread = None
        self.game_no = 0


    def _new_game(self, queue_controller: QueueController):
        # code partly copied from gupg/runner.py:Runner.run_game
        if not self.config.start_balancing or self.game_no % (len(self.config.controllers) + 1) == 0:
            arena = random.choice(self.config.arenas)
            random.shuffle(self.controllers_order)
            controllers = [(self.config.controllers[i - 1] if i != 0 else queue_controller) for i in self.controllers_order]
            game = games.Game(arena, controllers)
        else:
            self.controllers_order = self.controllers_order[1:] + [self.controllers_order[0]]
            controllers = [(self.config.controllers[i - 1] if i != 0 else queue_controller) for i in self.controllers_order]
            game = games.Game(
                self._last_arena,
                controllers,
                self._last_menhir_position,
                self._last_initial_positions
            )
        self._last_arena = game.arena.name
        self._last_menhir_position = game.arena.menhir_position
        self._last_initial_positions = game.initial_champion_positions
        self.game_no += 1
        return game
    

    @property
    def action_space(self):
        return gym.spaces.Discrete(3)
    
    @property
    def observation_space(self):
        return gym.spaces.Dict({"arena": DescriptionSpace(), "view": ViewSpace()})
    
    def _wait_and_get_observation(self) -> dict:
        knowledge = self.knowledge_queue.get(block=True, timeout=self.OBSERVATION_TIMEOUT)
        if (current_map := self.queue_controller.current_map) is None:
            raise RuntimeError("ThreadController has no current map!")
        return {"arena": current_map, "view": knowledge}

    def reset(self):
        if self.game_thread is not None:
            self.stop_event.set()
            self.game_thread.join()

        self.action_queue: queue.Queue[characters.Action] = queue.Queue(1)
        self.knowledge_queue: queue.Queue[ChampionKnowledge] = queue.Queue(1)
        self.queue_controller = QueueController(self.action_queue, self.knowledge_queue)
        new_game = self._new_game(self.queue_controller)
        self.stop_event = threading.Event()
        self.game_thread = threading.Thread(target=run_game, args=(new_game, self.stop_event))
        self.game_thread.start()

        return self._wait_and_get_observation(), {}

    def step(self, action: int):
        game_action = AgentAction2Action[AgentAction(action)]
        self.action_queue.put(game_action)
        
        obs = self._wait_and_get_observation()
        reward = 1 
        done = False # TODO
        truncated = False 
        info = {}
        return obs, reward, done, truncated, info

    def render(self, mode='human'):
        pass

    def close(self):
        self.stop_event.set()
        if self.game_thread is not None:
            self.game_thread.join()

register(
     id='GUPB-v0',
     entry_point='gupb.gym_env.gym_env:GUPBEnv',
)
