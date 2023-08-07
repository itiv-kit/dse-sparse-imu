import torch
from hpe_from_imu.configuration import Config
from hpe_from_imu.configuration import Constants as C
from hpe_from_imu.utils import full_to_reduced_SMPL_Pose, gaussian_noise
from pytorch3d.transforms.rotation_conversions import (axis_angle_to_matrix,
                                                       matrix_to_axis_angle)

IMU_mask = Config(C.config_path)["TP_joint_set"]["reduced"]


class IMUDatasetInwardsTransforms(object):
    def __init__(self):
        super().__init__()

    def __call__(self, acc, ori, pose):
        return acc, ori, pose

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class PoseAsRotMatrix(IMUDatasetInwardsTransforms):
    def __init__(self):
        super().__init__()

    def __call__(self, acc, ori, pose):
        pose = axis_angle_to_matrix(pose.view(-1, 24, 3))
        return acc, ori, pose


class PoseAsAxisAngle(IMUDatasetInwardsTransforms):
    def __init__(self):
        super().__init__()

    def __call__(self, acc, ori, pose):
        pose = matrix_to_axis_angle(pose)
        return acc, ori, pose


class ReducedPoseAsAxisAngle(IMUDatasetInwardsTransforms):
    def __init__(self):
        super().__init__()

    def __call__(self, acc, ori, pose):
        pose = matrix_to_axis_angle(
            pose.view(-1, 15, 3, 3)).flatten(-2)
        return acc, ori, pose


class ToReducedSMPL(IMUDatasetInwardsTransforms):
    def __init__(self):
        super().__init__()

    def __call__(self, acc, ori, pose):
        pose = full_to_reduced_SMPL_Pose(pose, IMU_mask)
        pose = pose.flatten(-3)
        return acc, ori, pose


class AddAccToPose(IMUDatasetInwardsTransforms):
    def __init__(self):
        super().__init__()

    def __call__(self, acc, ori, pose):
        return acc, ori, torch.cat((pose, acc), dim=1)


class AddNoise(IMUDatasetInwardsTransforms):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def __call__(self, acc, ori, pose):
        return gaussian_noise(acc, self.sigma), gaussian_noise(ori, self.sigma), pose


class DownsampleInput(IMUDatasetInwardsTransforms):
    def __init__(self, keep_nth=2):
        super().__init__()
        self._keep_nth = keep_nth

    def __call__(self, acc, ori, pose):
        return acc[::self._keep_nth], ori[::self._keep_nth], pose[::self._keep_nth]


class NormalizeWRTRoot(IMUDatasetInwardsTransforms):
    """
    Adapted from https://github.com/Xinyu-Yi/TransPose/blob/37be773ceceda49160717311a4f74a695817b8d3/utils.py
    """

    def __init__(self):
        super().__init__()
        self._acc_scale = 30

    def __call__(self, acc, ori, pose):
        glb_acc = acc.view(-1, 6, 3)
        glb_ori = ori.view(-1, 6, 3, 3)
        lcl_acc = torch.cat(
            (glb_acc[:, :5] - glb_acc[:, 5:],
             glb_acc[:, 5:]),
            dim=1).bmm(glb_ori[:, -1]) / self._acc_scale
        lcl_ori = torch.cat(
            (glb_ori[:, 5:].transpose(2, 3).matmul(glb_ori[:, :5]),
             glb_ori[:, 5:]),
            dim=1)
        return lcl_acc.flatten(start_dim=1), lcl_ori.flatten(start_dim=1), pose

class NormalizeSynthWRTRoot(IMUDatasetInwardsTransforms):
    """
    Adapted from https://github.com/Xinyu-Yi/TransPose/blob/37be773ceceda49160717311a4f74a695817b8d3/utils.py
    """

    def __init__(self):
        super().__init__()
        self._acc_scale = 30

    def __call__(self, acc, ori, pose):
        # calculate number of joints/vertices to flexible function
        n_joints = int(len(acc[0])/3)

        glb_acc = acc.view(-1, n_joints, 3)
        glb_ori = ori.view(-1, n_joints, 3, 3)
        lcl_acc = torch.cat(
            (glb_acc[:, :(n_joints-1)] - glb_acc[:, (n_joints-1):],
             glb_acc[:, (n_joints-1):]),
            dim=1).bmm(glb_ori[:, -1]) / self._acc_scale
        lcl_ori = torch.cat(
            (glb_ori[:, (n_joints-1):].transpose(2, 3).matmul(glb_ori[:, :(n_joints-1)]),
             glb_ori[:, (n_joints-1):]),
            dim=1)
        return lcl_acc.flatten(start_dim=1), lcl_ori.flatten(start_dim=1), pose
