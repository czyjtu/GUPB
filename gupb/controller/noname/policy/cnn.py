import torch as th 
import typing as tp


def ConvBlockNormalizeDrop(in_channels: int, n_filters: int, activation: tp.Literal["sigmoid", "relu"], p: float, gap: bool=False):
    activation_fun = {"relu": th.nn.ReLU, "sigmoid": th.nn.Sigmoid}[activation]
    return th.nn.Sequential(
        th.nn.Conv2d(in_channels, n_filters, 3, padding="same"),
        activation_fun(),
        th.nn.BatchNorm2d(n_filters),
        th.nn.Conv2d(n_filters, n_filters, 3, padding="same"),
        activation_fun(),
        th.nn.BatchNorm2d(n_filters),
        th.nn.AdaptiveAvgPool2d((1, 1)) if gap else th.nn.MaxPool2d(2),
        th.nn.Dropout2d(p)
    )

class CNNPolicy(th.nn.Module):
    def __init__(self, actions_num: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = ConvBlockNormalizeDrop(3, 8, "relu", 0.2)
        self.conv2 = ConvBlockNormalizeDrop(8, 32, "relu", 0.2, gap=True)
        self.linear = th.nn.Linear(32, actions_num)

    def forward(self, X: th.Tensor) -> th.Tensor:
        X = self.conv1(X)
        X = self.conv2(X)
        X = X.view(X.shape[0], -1)
        X = self.linear(X)
        return X
    