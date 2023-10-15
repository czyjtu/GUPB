import queue
import threading
import time
import numpy as np
import pytest 
import gym 

from gupb.controller import random
from  gupb.gym_env.gym_env import AgentAction, EnvConfig, QueueController
from gupb.model import characters
from gupb.model.characters import ChampionKnowledge

@pytest.fixture
def env():
    env = gym.make(
        "GUPB-v0", 
        config=EnvConfig(
            arenas=["lone_sanctum"], 
            controllers=[random.RandomController("Alice"), random.RandomController("Bob")]
        )
    )
    yield env 
    env.close()

@pytest.fixture 
def empty_knowledge():
    return ChampionKnowledge((0, 0), 0, {})


def test_env_build_successfully(env):
    assert env is not None
    assert env.reset() is not None



def test_reset_return_arena_and_view(env):
    obs, _ = env.reset()

    assert obs is not None
    match obs:
        case {"arena": _, "view": _}: 
            pass
        case _:
            assert False, obs
    

def test_observation_contains_actual_view(env):

    obs, _ = env.reset()

    assert obs is not None
    assert isinstance(obs["view"], ChampionKnowledge)


def test_if_queue_controller_returns_action_from_queue(empty_knowledge):
    action = characters.Action.ATTACK

    action_queue = queue.Queue()
    knowledge_queue = queue.Queue()

    controller = QueueController(action_queue, knowledge_queue)

    action_queue.put(action)
    assert controller.decide(empty_knowledge) == action


@pytest.mark.timeout(QueueController.ACTION_TIMEOUT + 1)
def test_if_queue_controller_raise_exception_on_empty_queue(empty_knowledge):
    action_queue = queue.Queue()
    knowledge_queue = queue.Queue()
    controller = QueueController(action_queue, knowledge_queue)
    with pytest.raises(queue.Empty):
        controller.decide(empty_knowledge)


@pytest.mark.timeout(QueueController.ACTION_TIMEOUT + 1)
def test_if_queue_controller_put_knowledge_into_queue(empty_knowledge):
    action = characters.Action.ATTACK

    action_queue = queue.Queue()
    knowledge_queue = queue.Queue()

    action_queue.put(action)
    controller = QueueController(action_queue, knowledge_queue)

    _ = controller.decide(empty_knowledge)
    assert knowledge_queue.get(block=False) == empty_knowledge


def test_if_queue_controller_works_from_different_thread(empty_knowledge):         
    def loop_thread(controller: QueueController, observations: list[ChampionKnowledge]):
        for o in observations:
            _ = controller.decide(o)
            

    action_queue = queue.Queue()
    knowledge_queue = queue.Queue()
    controller = QueueController(action_queue, knowledge_queue)

    o1 = ChampionKnowledge((0, 0), 1, {})
    o2 = ChampionKnowledge((0, 1), 1, {})
    o3 = ChampionKnowledge((1, 1), 1, {})

    thread = threading.Thread(target=loop_thread, args=(controller, [o1, o2, o3]))
    thread.start()

    assert knowledge_queue.get(block=True) == o1
    action_queue.put(characters.Action.ATTACK)
    assert knowledge_queue.get(block=True) == o2
    action_queue.put(characters.Action.ATTACK)
    assert knowledge_queue.get(block=True) == o3
    action_queue.put(characters.Action.ATTACK)
    thread.join()
