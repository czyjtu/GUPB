import pytest 
import torch as th 
from gupb.controller.noname.policy import CNNEncoder

@pytest.fixture 
def network():
    return CNNEncoder(5).eval()

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
    net = CNNEncoder(latent_size=actions_num)
    assert net(th.rand(1, 3, 10, 10)).shape[1] == actions_num
