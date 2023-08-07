import os

import torch
from hpe_from_imu.configuration import Config
from hpe_from_imu.configuration import Constants as C
from hpe_from_imu.dataloader.dataset_transforms_in import \
    IMUDatasetInwardsTransforms
from torch.utils.data import DataLoader, Dataset

paths = Config(C.config_path)["paths"]
sub_sensor_config = []


def getDataLoader(dataset: Dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)


def getIMUDataLoader(file_path: str, batch_size=1, transforms=None):
    data = torch.load(file_path)
    if len(sub_sensor_config) > 0:
        data_reduced = {}
        data_reduced["ori"] = list(map(lambda x: x[:,sub_sensor_config,:,:], data["ori"]))
        data_reduced["acc"] = list(map(lambda x: x[:,sub_sensor_config,:], data["acc"]))
        data_reduced["pose"] = data["pose"]
        dataset = IMU_Dataset(data_reduced, transforms)
    else:
        dataset = IMU_Dataset(data, transforms)
    return getDataLoader(dataset, batch_size=batch_size)


def getGeneralDataLoader(file_path: str, batch_size=1):
    data = torch.load(file_path)
    dataset = General_Dataset(data)
    return getDataLoader(dataset, batch_size=batch_size)


def getExampleDataLoader(file: str, transforms=None):
    data = torch.load(os.path.join(paths["example_dir"], file + "-example.pt"))
    dataset = IMU_Dataset(data, transforms)
    return getDataLoader(dataset, batch_size=1)


class General_Dataset(Dataset):
    """
    General Dataset class that takes an object with indexes input and output. Based on Dataset from PyTorch and can be used for Dataloaders.
    """

    def __init__(self, data):
        super().__init__()
        self.input = data["input"]
        self.output = data["output"]

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx].flatten(start_dim=1), self.output[idx].flatten(start_dim=1)


class IMU_Dataset(Dataset):
    """
    IMU Dataset class that takes an object with indexes acc, ori and pose. Based on Dataset from PyTorch and can be used for Dataloaders.
    Pass a list of IMUDatasetTransformations that will be used to transform acc, ori and pose.
    """

    def __init__(self, data, transforms: list[IMUDatasetInwardsTransforms] = None):
        super().__init__()
        self.acc = data["acc"]
        self.ori = data["ori"]
        self.pose = data["pose"]
        self.transforms = transforms

    def __len__(self):
        return len(self.acc)

    def __getitem__(self, idx):
        acc = self.acc[idx].flatten(start_dim=1)
        ori = self.ori[idx].flatten(start_dim=1)
        pose = self.pose[idx]
        if self.transforms is not None:
            for transform in self.transforms:
                acc, ori, pose = transform(acc, ori, pose)
        return torch.cat((acc, ori), dim=1), pose
