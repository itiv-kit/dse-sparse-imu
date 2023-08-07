from hpe_from_imu.dataloader import (IMUDatasetInwardsTransforms,
                                     IMUDatasetOutwardsTransforms)
from torch.nn import Module


class BaseIMUNet(Module):
    """
    A base IMU model that implements the functionality to set transformations for datasets.
    """

    def __init__(self, name, layers, transforms_in={}, transforms_out=[]):
        super().__init__()
        self._name = name
        self.layers: Module = layers
        self._transforms_in: dict[str,
                                  IMUDatasetInwardsTransforms] = transforms_in
        self._transforms_out: list[IMUDatasetOutwardsTransforms] = transforms_out

    @property
    def transforms_in(self):
        return self._transforms_in

    @property
    def transforms_out(self):
        return self._transforms_out

    def forward(self, x):
        return self.layers.forward(x)

    def __repr__(self) -> str:
        return f"{self._name}"
