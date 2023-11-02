import queue 

from gupb import controller
from gupb.model import arenas, characters
from gupb.model.characters import ChampionKnowledge


class QueueController(controller.Controller):
    ACTION_TIMEOUT = 300 # TODO: make it configurable, change to higher value during traning

    def __init__(self, action_queue, knowledge_queue) -> None:
        self.action_queue: queue.Queue = action_queue
        self.knowledge_queue: queue.Queue = knowledge_queue
        self.current_map = None 

    @property
    def name(self):
        return "ThreadController"

    @property 
    def preferred_tabard(self):
        return characters.Tabard.PINK

    def decide(self, knowledge: ChampionKnowledge) -> characters.Action:
        self.knowledge_queue.put(knowledge, block=True, timeout=self.ACTION_TIMEOUT)
        return self.action_queue.get(block=True, timeout=self.ACTION_TIMEOUT)  

    def praise(self, score: int) -> None:
        pass

    def reset(self, arena_description: arenas.ArenaDescription) -> None:
        self.current_map = arena_description.name