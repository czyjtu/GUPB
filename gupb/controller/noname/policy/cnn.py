import torch as th 

class CNNPolicy(th.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, X: th.Tensor) -> th.Tensor:
        return th.rand(X.shape[0], 5)
    