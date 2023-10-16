import queue
import threading
import time
import numpy as np
import pytest 
import gym 

from gupb.controller import random
from  gupb.gym_env.gym_env import AgentAction, EnvConfig, GUPBEnv, QueueController
from gupb.model import characters
from gupb.model.characters import ChampionKnowledge


@pytest.fixture
def env():
    env = gym.make(
        "GUPB-v0", 
        config=EnvConfig(
            arenas=["lone_sanctum", "ordinary_chaos"], 
            controllers=[random.RandomController("Alice"), random.RandomController("Bob")]
        )
    )
    yield env 
    env.close()


@pytest.fixture
def balancing_env():
    env = gym.make(
        "GUPB-v0", 
        config=EnvConfig(
            arenas=["lone_sanctum", "ordinary_chaos"], 
            controllers=[random.RandomController("Alice"), random.RandomController("Bob")],
            start_balancing=True
        )
    )
    yield env 
    env.close()


@pytest.fixture 
def empty_knowledge():
    return ChampionKnowledge((0, 0), 0, {})


def test_env_close_the_thread(env):
    assert threading.active_count() == 1
    env.reset()
    assert threading.active_count() == 2
    env.close()
    assert threading.active_count() == 1


def test_if_reset_joins_the_thread(env):
    assert threading.active_count() == 1

    env.reset()
    assert threading.active_count() == 2

    env.reset()
    env.reset()
    assert threading.active_count() == 2

    env.step(AgentAction.TURN_LEFT.value)
    env.step(AgentAction.TURN_LEFT.value)
    assert threading.active_count() == 2

    env.step(AgentAction.TURN_LEFT.value)
    env.reset()
    assert threading.active_count() == 2

    env.close()
    assert threading.active_count() == 1


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


@pytest.mark.parametrize("action", [AgentAction.TURN_LEFT, AgentAction.TURN_RIGHT])
def test_if_turn_action_takes_effect(env, action):
    obs1, _ = env.reset()
    obs2, *_ = env.step(action.value)
    obs3, *_ = env.step(action.value)

    coord1 = obs1["view"]
    coord2 = obs2["view"]
    coord3 = obs3["view"]
    assert coord2 != coord3 != coord1

@pytest.mark.parametrize("action", [AgentAction.STEP_FORWARD])
def test_if_forward_action_takes_effect(env, action):
    max_tries = 5
    # loop until terrain passable 
    for _ in range(max_tries):
        obs1, _ = env.reset()
        current_coord = obs1["view"].position
        next_coord = current_coord + obs1["view"].visible_tiles[current_coord].character.facing.value
        next_tile = obs1["view"].visible_tiles[next_coord]
        if next_tile.type in ["land", "menhir"] and next_tile.character is None:
            break

    obs2, *_ = env.step(action.value)
    assert obs2["view"].position == next_coord


def test_if_balancing_works_properly(balancing_env: GUPBEnv):
    # we have 3 controllers, therefore each map is played 3 times with different controllers order
    obs11, _ = balancing_env.reset()
    controllers_order11 = balancing_env.controllers_order.copy()
    obs12, _ = balancing_env.reset()
    controllers_order12 = balancing_env.controllers_order.copy()
    obs13, _ = balancing_env.reset()
    controllers_order13 = balancing_env.controllers_order.copy()
    assert obs11["arena"] == obs12["arena"]  == obs13["arena"]
    assert controllers_order11 != controllers_order12 != controllers_order13

    # now we hould start new random map, and play it 3 times with different controllers order 
    obs21, _ = balancing_env.reset()
    controllers_order21 = balancing_env.controllers_order.copy()
    assert controllers_order13 != controllers_order21
    obs22, _ = balancing_env.reset()
    controllers_order22 = balancing_env.controllers_order.copy()
    obs23, _ = balancing_env.reset()
    controllers_order23 = balancing_env.controllers_order.copy()
    assert obs21["arena"] == obs22["arena"]  == obs23["arena"]
    assert controllers_order21 != controllers_order22 != controllers_order23
