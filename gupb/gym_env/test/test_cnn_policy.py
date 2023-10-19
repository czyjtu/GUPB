import pytest 
import torch as th 
from gupb.controller.noname.policy import CNNPolicy

def test_cnn_network_returns_tensor():
    network = CNNPolicy()
    observation = th.rand(1, 3, 10, 10)
    result = network(observation)
    assert isinstance(result, th.Tensor)
    