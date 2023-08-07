import torch
from hpe_from_imu.utils import gaussian_noise


class GaussianNoise(torch.nn.Module):
    """
    Gaussian noise regularizer.

    Based on https://discuss.pytorch.org/t/where-is-the-noise-layer-in-pytorch/2887/4

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach

    def forward(self, x: torch.Tensor):
        if self.training and self.sigma != 0:
            return gaussian_noise(x, self.sigma, self.is_relative_detach)
        return x
