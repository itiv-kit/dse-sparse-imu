import torch
import numpy as np
import cv2
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix
from torch.nn.functional import normalize


def angle_between_vectors(a: torch.Tensor, b: torch.Tensor):
    """
    Calculate the angle in degress between two vectors. Calculates the dot product of the normalized vectors.

    Args:
        a (torch.Tensor): Tensor of shape (A, B, C)
        b (torch.Tensor): Tensor of shape (A, B, C)

    Returns:
        torch.Tensor: Tensor of angles between a and b shape (A, B)
    """
    return torch.clamp(torch.einsum('bnm,bnm->bn', normalize(a, dim=2), normalize(b, dim=2)), min=-1, max=1).acos().rad2deg()


def angle_between_rotations_axis_angles(a: torch.Tensor, b: torch.Tensor):
    """
    Calculates the angle in degress between two rotations in axis-angle form.

    Args:
        a (torch.Tensor): Tensor of shape (..., 3)
        b (torch.Tensor): Tensor of shape (..., 3)

    Returns:
        torch.Tensor: Tensor of angles between a and b. Output has shape (...)
    """
    offset_matrix = torch.matmul(axis_angle_to_matrix(
        a).transpose(-1, -2), axis_angle_to_matrix(b))
    return torch.clamp(((torch.einsum("bnii", offset_matrix) - 1)*0.5), min=-1, max=1).acos().rad2deg()

    #seq_len, n_joints, dof = a.shape[0], a.shape[1], a.shape[2]
    #assert n_joints == 24, 'unexpected number of joints'
    #assert dof == 3, 'unexpected number of degrees of freedom'
    #p1 = axis_angle_to_matrix(a) # predicted
    #p2 = axis_angle_to_matrix(b) # targed
    #r1 = torch.reshape(p1, [-1,3,3])
    #r2 = torch.reshape(p2, [-1,3,3])
    #r2t = torch.transpose(r2, 2, 1)
    #r = torch.matmul(r1, r2t)
    #angles =[]
    #for i in range(r1.shape[0]):
    #    aa, _ = cv2.Rodrigues(r[i].detach().cpu().numpy())
    #    angles.append(np.rad2deg(np.linalg.norm(aa)))
    #return torch.reshape(torch.tensor(angles), [seq_len, n_joints])


def calc_jerk(x: torch.Tensor, steps_per_second=60):
    """
    Calculates the jerk for given sequence. Jerk is the time derivative of acceleration or third time derivative of position.
    Uses third order backward difference as approximation formula.

    S: Sequence length

    Args:
        x (torch.Tensor): Tensor of sequence to analyse. Should have shape (S , ... ).
        steps_per_second (int, optional): Sample rate of the sequence. Defaults to 60 as in 60 samples per second.

    Returns:
        torch.Tensor: Jerk of given sequence. Has shape ( (S-3) * ...)
    """
    return ((x[3:] - 3 * x[2:-1] + 3 * x[1:-2] - x[:-3]) * (steps_per_second ** 3))
