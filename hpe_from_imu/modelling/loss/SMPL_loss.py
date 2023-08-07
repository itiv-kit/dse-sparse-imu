import torch.nn as nn
from hpe_from_imu.configuration import Config
from hpe_from_imu.configuration import Constants as C
from hpe_from_imu.evaluation import PoseEvaluator
from hpe_from_imu.utils import reduced_to_full_SMPL_pose
from pytorch3d.transforms import matrix_to_axis_angle

IMU_mask = Config(C.config_path)["TP_joint_set"]["reduced"]


class SMPLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        n_acc = len(y_pred[0,0])-(len(IMU_mask)*9)
        y_ = self._to_full_axis_angle(y_pred[:, :, :-n_acc])
        y = self._to_full_axis_angle(y_true[:, :, :-n_acc])
        evaluator = PoseEvaluator()
        errs = evaluator.eval(y_, y)
        return errs[:, 0].mean()

    def __str__(self):
        return "SMPLLoss"

    def _to_full_axis_angle(self, pose):
        return matrix_to_axis_angle(reduced_to_full_SMPL_pose(pose, IMU_mask))
