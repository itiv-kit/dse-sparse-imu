import torch
from hpe_from_imu.configuration import Config
from hpe_from_imu.configuration import Constants as C
from hpe_from_imu.utils.tensor_utils import expand_to_bigger_tensor
from pytorch3d.transforms.rotation_conversions import (axis_angle_to_matrix,
                                                       matrix_to_axis_angle)
from smplx import SMPL

paths = Config(C.config_path)["paths"]


def full_to_reduced_SMPL_Pose(input: torch.Tensor, mask: torch.Tensor):
    """
    Transforms a tensor from a full SMPL pose to a reduced pose that only contains the SMPL joints specified in mask.
    Requires rotation matrix as rotation representation.

    B: Batch size
    S: Sequence length

    Args:
        input (torch.Tensor): Full SMPL pose input tensor of shape (B, S, 24, 3, 3)
        mask (torch.Tensor | numpy.array): Mask that contains index of SMPL joints to include. Allowed joints are between 0 and 23.

    Returns:
        torch.Tensor: Output tensor that only contains masked SMPL joints. Output tensor of shape (B, S, len(mask), 3, 3)
    """
    return input[..., mask, :, :]


def reduced_to_full_SMPL_pose(input: torch.Tensor, mask: torch.Tensor):
    """
    Transforms a tensor from reduced SMPL pose to full SMPL pose.

    B: Batch size
    S: Sequence length

    Args:
        input (torch.Tensor): Input tensor of shape (B, S, len(mask), 3, 3) that should be expanded to include all SMPL joints.
        mask (torch.Tensor | numpy.array): Mask that contains the SMPL joints included in input.

    Returns:
        torch.Tensor: Output tensor that contains the input at specified indexes and identity matrices where no input was given.
        Output tensor of shape (B, S, 24, 3, 3)
    """
    view = input.view(input.size(0), input.size(1), len(mask), 3, 3)
    return expand_to_bigger_tensor(view, 24, mask)


def SMPL_local_to_global(smpl_local):
    """
    Converts local SMPL rotations into global rotations by "unrolling" the kinematic chain.

    Based on https://github.com/eth-ait/dip18/blob/917e6199ffa5eabed2cc8790c6bcd8337f369c3e/train_and_eval/utils.py#L214

    Args:
        smpl_local (torch.Tensor): tensor of rotation matrices of SMPL pose parameters of shape (..., 24, 3)

    Returns:
        torch.Tensor: The global rotations as a tensor of the same shape as the input (..., 24, 3).
    """

    SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7,
                    8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]

    rots = axis_angle_to_matrix(smpl_local)
    out = torch.zeros_like(rots)
    for j in range(24):
        if SMPL_PARENTS[j] < 0:
            out[..., j, :, :] = rots[..., j, :, :]
        else:
            parent_rot = out[..., SMPL_PARENTS[j], :, :]
            local_rot = rots[..., j, :, :]
            out[..., j, :, :] = parent_rot @ local_rot
    return matrix_to_axis_angle(out)


class BodyModel(SMPL):
    """
    Opinionized wrapper around an SMPL layer.
    """

    def __init__(self, model_path=paths["SMPL_male"]):
        """
        Initializes the body model with path and device.

        Args:
            model_path(str, optional): Path for SMPL model to use.
            device(torch.Device, optional): Device to use. Defaults to torch.device("cuda") if available.
        """
        super(BodyModel, self).__init__(model_path=model_path)

    def forward_kinematics(self, pose: torch.Tensor, shape=torch.Tensor([0.0] * 10), translation: torch.Tensor = None):
        """
        Calculates the SMPL mesh and joint positions for the given pose parameters and shape parameters

        S: Sequence length

        Args:
            pose (torch.Tensor): SMPL pose tensor of shape (S, 24, 3)
            shape (torch.Tensor, optional): SMPL shape tensor of shape (10). Defaults to torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).
            translation (torch.Tensor, optional): Translation tensor of shape (S, 3). Defaults to None.

        Returns:
            tuple(torch.Tensor, torch.Tensor): The joints of shape (S, 24, 3) and vertices of shape (S, 6890, 3).
        """
        betas = torch.vstack([shape.to(device=pose.device)] * len(pose))
        output = self.forward(betas=betas,
                              body_pose=pose[:, 1:, :],
                              global_orient=pose[:, :1, :],
                              transl=translation)
        return output.joints[:, :24], output.vertices
