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

class CNNEncoder(th.nn.Module):
    def __init__(self, channels: int, latent_size: int, arch: list[int], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        convs = []
        for n_filters in arch:
            convs.append(ConvBlockNormalizeDrop(channels, n_filters, "relu", 0.05))
            channels = n_filters
        convs.append(
            ConvBlockNormalizeDrop(channels, latent_size, "relu", 0, gap=True)
        )
        self.convs = th.nn.Sequential(*convs)

    def forward(self, X: th.Tensor) -> th.Tensor:
        X = self.convs(X)
        X = X.view(X.shape[0], -1)
        return X
    