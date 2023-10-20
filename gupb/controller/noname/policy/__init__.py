from gupb.controller.noname.policy.cnn import CNNEncoder
import torch as th 


class ActorCriticTorchPolicy(th.nn.Module):
    def __init__(self, feature_extractor: th.nn.Module, actor: th.nn.Module, critic: th.nn.Module, action_net: th.nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.actor = actor
        self.critic = critic
        self.action_net = action_net

    def forward(self, X: th.Tensor) -> th.Tensor:
        features = self.feature_extractor(X)
        latent_pi = self.actor(features)
        distribution_probs = self.action_net(latent_pi)
        action = th.argmax(distribution_probs, dim=1)
        return action