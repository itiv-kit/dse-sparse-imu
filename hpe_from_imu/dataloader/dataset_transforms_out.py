from hpe_from_imu.configuration import Config
from hpe_from_imu.configuration import Constants as C
from hpe_from_imu.utils import reduced_to_full_SMPL_pose
from pytorch3d.transforms.rotation_conversions import (axis_angle_to_matrix,
                                                       matrix_to_axis_angle)

IMU_mask = Config(C.config_path)["TP_joint_set"]["reduced"]


class IMUDatasetOutwardsTransforms(object):
    def __init__(self):
        super().__init__()

    def __call__(self, pose):
        return pose

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


class RemoveAcc(IMUDatasetOutwardsTransforms):
    def __init__(self):
        super().__init__()

    def __call__(self, pose):
        n_acc = len(pose[0,0])-(len(IMU_mask)*9)
        return pose[:, :, :-n_acc]


class ToFullSMPL(IMUDatasetOutwardsTransforms):
    def __init__(self):
        super().__init__()

    def __call__(self, pose):
        return reduced_to_full_SMPL_pose(pose, IMU_mask)


class ToAxisAngle(IMUDatasetOutwardsTransforms):
    def __init__(self):
        super().__init__()

    def __call__(self, pose):
        return matrix_to_axis_angle(pose)


class ToReducedRotMat(IMUDatasetOutwardsTransforms):
    def __init__(self):
        super().__init__()

    def __call__(self, pose):
        return axis_angle_to_matrix(pose.view(pose.size(0), pose.size(1), 15, 3))


class ToFullRotMat(IMUDatasetOutwardsTransforms):
    def __init__(self):
        super().__init__()

    def __call__(self, pose):
        return axis_angle_to_matrix(pose.view(pose.size(0), pose.size(1), 24, 3))
