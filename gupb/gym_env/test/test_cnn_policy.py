import pytest 
import torch as th 
from gupb.controller.noname.policy import ActorCriticTorchPolicy, CNNEncoder
from gupb.gym_env.agent.actor_critic import CustomActorCriticPolicy, CustomFeatureExtractor, extract_atoms_from_ppo_model, PPO2TorchPolicy
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
            actor_arch=[5],
            critic_arch=[5],
        ),
    )
    model.learn(total_timesteps=5)
    # Enjoy trained agent
    obs, _ = env.reset()
    for i in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, truncated, info = env.step(action)


def test_trained_policy_is_loaded_from_file_properly(tmp_path):
    features_dim = 64 # it is latent size for CNNEncoder    

    env = MockEnv(gym.spaces.Box(low=0, high=1, shape=(3, 10, 10)), gym.spaces.Discrete(3))
    model = PPO(
        CustomActorCriticPolicy,
        env,
        policy_kwargs=dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=features_dim),
            actor_arch=[5],
            critic_arch=[5],
        ),
    )
    model.learn(total_timesteps=5)
    model.save(tmp_path / "ppo_custom")

    model2 = PPO.load(tmp_path / "ppo_custom")
    obs, _ = env.reset()
    assert model.predict(obs, deterministic=True) == model2.predict(obs, deterministic=True)


def test_wether_custom_networks_are_extracted_properly_from_ppo():
    features_dim = 64 # it is latent size for CNNEncoder    

    env = MockEnv(gym.spaces.Box(low=0, high=1, shape=(3, 10, 10)), gym.spaces.Discrete(3))
    model = PPO(
        CustomActorCriticPolicy,
        env,
        policy_kwargs=dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=features_dim),
            actor_arch=[5],
            critic_arch=[5],
        ),
    )
    model.learn(total_timesteps=5)
    
    encoder, actor, critic, action_net = extract_atoms_from_ppo_model(model)
    assert isinstance(encoder, CNNEncoder)
    assert isinstance(actor, th.nn.Module)
    assert isinstance(critic, th.nn.Module)
    assert isinstance(action_net, th.nn.Module)


def test_wether_torch_actor_critic_policy_returns_the_same_action_as_ppo_policy():
    features_dim = 64 # it is latent size for CNNEncoder    
    env = MockEnv(gym.spaces.Box(low=0, high=1, shape=(3, 10, 10)), gym.spaces.Discrete(3))
    model = PPO(
        CustomActorCriticPolicy,
        env,
        policy_kwargs=dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=features_dim),
            actor_arch=[5],
            critic_arch=[5],
        ),
    )
    model.learn(total_timesteps=5)
    torchPolicy = PPO2TorchPolicy(model)

    obs, _ = env.reset()
    pred, _ = model.predict(obs, deterministic=True)
    assert pred == torchPolicy(th.Tensor(obs).unsqueeze(0)).numpy()


@pytest.mark.parametrize("latent_size", [1, 10, 256])
def test_if_actor_network_latent_size_is_independent_of_action_space(latent_size):
    features_dim = 64 # it is latent size for CNNEncoder    
    env = MockEnv(gym.spaces.Box(low=0, high=1, shape=(3, 10, 10)), gym.spaces.Discrete(3))
    model = PPO(
        CustomActorCriticPolicy,
        env,
        policy_kwargs=dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=features_dim),
            actor_arch=[latent_size],
            critic_arch=[latent_size],
        ),
    )
    model.learn(total_timesteps=5)
    encoder, actor, critic, action_net = extract_atoms_from_ppo_model(model)
    assert actor(th.rand(1, features_dim)).shape[1] == latent_size
    assert critic(th.rand(1, features_dim)).shape[1] == latent_size

@pytest.mark.parametrize("critic_arch", [[1, 10, 256], [512, 128, 128, 64]])
@pytest.mark.parametrize("actor_arch", [[1, 10, 256], [512, 128, 128, 64]])
def test_if_actorcritic_network_architecture_is_applied(critic_arch, actor_arch):
    features_dim = 64 # it is latent size for CNNEncoder    
    env = MockEnv(gym.spaces.Box(low=0, high=1, shape=(3, 10, 10)), gym.spaces.Discrete(3))
    model = PPO(
        CustomActorCriticPolicy,
        env,
        policy_kwargs=dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=features_dim),
            actor_arch=actor_arch,
            critic_arch=critic_arch,
        ),
    )
    encoder, actor, critic, action_net = extract_atoms_from_ppo_model(model)
    assert isinstance(actor, th.nn.Sequential)
    assert isinstance(critic, th.nn.Sequential)
    actor_layers = [layer for layer in actor if isinstance(layer, th.nn.Linear)]
    critic_layers = [layer for layer in critic if isinstance(layer, th.nn.Linear)]
    assert len(actor_layers) == len(actor_arch)
    assert len(critic_layers) == len(critic_arch)
    assert [layer.out_features for layer in actor_layers]== actor_arch
    assert [layer.out_features for layer in critic_layers] == critic_arch


def test_saving_and_loading_torch_policy(tmp_path):
    features_dim = 64 # it is latent size for CNNEncoder    
    env = MockEnv(gym.spaces.Box(low=0, high=1, shape=(3, 10, 10)), gym.spaces.Discrete(3))
    model = PPO(
        CustomActorCriticPolicy,
        env,
        policy_kwargs=dict(
            features_extractor_class=CustomFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=features_dim),
            actor_arch=[1, 2, 3],
            critic_arch=[1, 2, 3],
        ),
    )
    torchPolicy = PPO2TorchPolicy(model).eval()
    th.save(torchPolicy, tmp_path / "test_torch_policy")
    loaded_policy = th.load(tmp_path / "test_torch_policy")

    obs, _ = env.reset()
    assert torchPolicy(th.Tensor(obs).unsqueeze(0)).numpy() == loaded_policy(th.Tensor(obs).unsqueeze(0)).numpy()
    for key, value in torchPolicy.state_dict().items():
        assert (value == loaded_policy.state_dict()[key]).all(), key
