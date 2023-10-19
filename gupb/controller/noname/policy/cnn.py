import torch as th 

class CNNPolicy(th.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def forward(self, observation):
        return th.rand(1, 1, 5)
    