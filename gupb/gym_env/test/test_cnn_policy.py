import pytest 
import torch as th 
from gupb.controller.noname.policy import CNNEncoder
from gupb.gym_env.agent.actor_critic import CustomActorCriticPolicy, CustomFeatureExtractor
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from gupb.gym_env.agent.actor_critic import CustomNetwork
import gymnasium as gym 


class MockEnv(gym.Env):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *args, **kwargs):
        return self.observation_space.sample(), {}
    
    def step(self, action):
        return th.Tensor(self.observation_space.sample()), 0, False, False, {}
    
    def render(self, mode="human"):
        pass
    


@pytest.fixture 
def network():
    return CNNEncoder(3, 5).eval()

def test_cnn_network_returns_tensor(network):
    observation = th.rand(1, 3, 10, 10)
    result = network(observation)
    assert isinstance(result, th.Tensor)
    

@pytest.mark.parametrize("batches", [1, 32, 64, 128])
def test_cnn_network_returns_correct_num_of_batches(batches, network):
    observation = th.rand(batches, 3, 10, 10)
    result = network(observation)
    assert result.shape[0] == batches


@pytest.mark.parametrize("X", [th.rand(1, 3, 10, 10), th.rand(32, 3, 10, 10), th.rand(64, 3, 10, 10), th.rand(128, 3, 10, 10)])
def test_cnn_returns_same_actions_for_same_inputs_and_different_for_different(X, network):
    Y = th.rand_like(X)
    result_X1 = network(X)
    result_X2 = network(X)
    result_Y = network(Y)
    assert th.all(result_X1 == result_X2)
    assert th.any(result_X1 != result_Y)

@pytest.mark.parametrize("actions_num", [1, 5, 10])
def test_network_output_can_be_adjusted(actions_num):
    net = CNNEncoder(3, latent_size=actions_num)
    assert net(th.rand(1, 3, 10, 10)).shape[1] == actions_num


def test_custom_feature_extractor_can_be_used_in_PPO():
    features_dim = 64 # it is latent size for CNNEncoder

    env = MockEnv(gym.spaces.Box(low=0, high=1, shape=(3, 10, 10)), gym.spaces.Discrete(3))
    model = PPO(
        CustomActorCriticPolicy,
        env,
        policy_kwargs=dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=features_dim),
        ),
    )
    model.learn(total_timesteps=5)
    # Enjoy trained agent
    obs, _ = env.reset()
    for i in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, truncated, info = env.step(action)
    env.close()
