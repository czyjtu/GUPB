import pytest 
import torch as th 
from gupb.controller.noname.policy import CNNPolicy

def test_cnn_network_returns_tensor():
    network = CNNPolicy()
    observation = th.rand(1, 3, 10, 10)
    result = network(observation)
    assert isinstance(result, th.Tensor)
    

@pytest.mark.parametrize("batches", [1, 32, 64, 128])
def test_cnn_network_returns_correct_num_of_batches(batches):
    network = CNNPolicy()
    observation = th.rand(batches, 3, 10, 10)
    result = network(observation)
    assert result.shape[0] == batches

