from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from gupb.controller.noname.policy import CNNEncoder, ActorCriticTorchPolicy


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # observation.shape == CxHxW
        self.cnn_encoder = CNNEncoder(observation_space.shape[0], features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.cnn_encoder(observations)


class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        actor_arch: list[int],
        critic_arch: list[int],
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions

        # Policy network
        self.actor_arch = [feature_dim] + actor_arch
        self.critic_arch = [feature_dim] + critic_arch

        # Policy network
        policy_layers = []
        for i in range(len(self.actor_arch) - 1):
            policy_layers.append(nn.Linear(self.actor_arch[i], self.actor_arch[i+1]))
            policy_layers.append(nn.ReLU())
        self.policy_net = th.nn.Sequential(*policy_layers)

        # Value network
        critic_layers = []
        for i in range(len(self.critic_arch) - 1):
            critic_layers.append(nn.Linear(self.critic_arch[i], self.critic_arch[i+1]))
            critic_layers.append(nn.ReLU())
        self.value_net = th.nn.Sequential(*critic_layers)
    
    @property 
    def latent_dim_pi(self) -> int:
        return self.actor_arch[-1]
    
    @property
    def latent_dim_vf(self) -> int:
        return self.critic_arch[-1]

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        actor_arch: list[int],
        critic_arch: list[int],
        *args,
        **kwargs,
    ):
        self.actor_net_arch = actor_arch
        self.critic_net_arch = critic_arch
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

        assert isinstance(self.action_space, spaces.Discrete), (
            "This policy network only works "
            "with spaces.Discrete action space (spaces.Box is not supported)"
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim, self.actor_net_arch, self.critic_net_arch)


def extract_atoms_from_ppo_model(model: PPO) -> tuple[CNNEncoder, th.nn.Module, th.nn.Module, th.nn.Module]:
    # encoder -> actor -> actor_latent -> action_net -> action_distribution -> action
    #        -> critic -> critic_latent -> value_net -> value
    encoder = model.policy.features_extractor.cnn_encoder
    actor = model.policy.mlp_extractor.policy_net
    critic = model.policy.mlp_extractor.value_net
    action_net = model.policy.action_net
    return encoder, actor, critic, action_net

def PPO2TorchPolicy(model: PPO, eval=True) -> ActorCriticTorchPolicy:
    encoder, actor, critic, action_net = extract_atoms_from_ppo_model(model)
    return ActorCriticTorchPolicy(encoder, actor, critic, action_net).eval()
