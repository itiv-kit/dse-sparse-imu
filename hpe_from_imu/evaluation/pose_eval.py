from builtins import staticmethod
import os
import csv
from ntpath import join
import torch
import numpy as np
from hpe_from_imu.configuration import Config
from hpe_from_imu.configuration import Constants as C
from hpe_from_imu.utils import (BodyModel, SMPL_local_to_global,
                                angle_between_rotations_axis_angles, calc_jerk)
from pytorch3d.transforms.rotation_conversions import (axis_angle_to_matrix,
                                                       matrix_to_axis_angle)

conf = Config(C.config_path)
paths = conf["paths"]
ignored_joint_set = conf["TP_joint_set"]["ignored"]


class PoseEvaluator():
    """
    Evaluates the prediction of SMPL sequences against ground truth in five criteria.
    The criteria are SIP error (hip and shoulder position error), global joint angle error, joint position error, mesh error and jerk.
    These are the same criteria as used in https://github.com/Xinyu-Yi/TransPose/blob/d37d617bbee044e5c1ad2e853f883b1001a5f87b/articulate/evaluator.py#L269
    """

    def __init__(self, model_path=paths["SMPL_male"], device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Initializes the pose evaluator with the SMPL model path and device.

        Args:
            model_path(str, optional): Path for SMPL model to use.
            device(torch.Device, optional): Device to use. Defaults to torch.device("cuda") if available.
        """
        self.body_model = BodyModel(model_path=model_path).to(device)
        self.device = device

    def eval(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Evaluates the prediction of SMPL sequences against ground truth in five criteria.

        B: Batch size
        S: Sequence length

        Args:
            y_pred (torch.Tensor): SMPL predicted pose tensor that can be reshaped to (B * S * 24 * 3)
            y_true (torch.Tensor): SMPL true pose tensor that can be reshaped to (B * S * 24 * 3)

        Returns:
            torch.Tensor: Returns a tensor of shape (5 * 2) with the SIP error, global angle error, joint position error, mesh error and jerk. Mean and standard deviation of the errors are returned.
        """
        y_pred = matrix_to_axis_angle(y_pred) 
        y_true = matrix_to_axis_angle(y_true)
        y_pred = y_pred.to(self.device).view(-1, 24, 3)
        y_true = y_true.to(self.device).view(-1, 24, 3)
        y_pred[:, ignored_joint_set] = torch.Tensor(
            [0.0, 0.0, 0.0]).to(self.device)
        y_true[:, ignored_joint_set] = torch.Tensor(
            [0.0, 0.0, 0.0]).to(self.device) 

        y_pred_joints, y_pred_vertices = self.body_model.forward_kinematics(
            y_pred)
        y_true_joints, y_true_vertices = self.body_model.forward_kinematics(
            y_true)

        offset = (y_true_joints[:, 0] - y_pred_joints[:, 0]).unsqueeze(1)

        global_angle_err = angle_between_rotations_axis_angles(
            SMPL_local_to_global(y_pred), SMPL_local_to_global(y_true))

        joint_mask = torch.tensor([1, 2, 16, 17])
        masked_global_angle_err = global_angle_err[:, joint_mask]

        joint_err = (y_pred_joints + offset - y_true_joints).norm(dim=2) * 100

        vertices_err = (y_pred_vertices + offset -
                        y_true_vertices).norm(dim=2) * 100

        jerk = calc_jerk(y_pred_joints).norm(dim=2) * 0.01

        return torch.tensor([[masked_global_angle_err.mean(), masked_global_angle_err.std(dim=0).mean()],
                            [global_angle_err.mean(), global_angle_err.std(dim=0).mean()],
                            [joint_err.mean(), joint_err.std(dim=0).mean()],
                            [vertices_err.mean(), vertices_err.std(dim=0).mean()],
                            [jerk.mean(), jerk.std(dim=0).mean()], ])

    def eval_joints(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Evaluates the prediction of SMPL sequences against ground truth in two criteria for a set of 15 joints.

        B: Batch size
        S: Sequence length

        Args:
            y_pred (torch.Tensor): SMPL predicted pose tensor that can be reshaped to (B * S * 24 * 3)
            y_true (torch.Tensor): SMPL true pose tensor that can be reshaped to (B * S * 24 * 3)

        Returns:
            torch.Tensor: Returns a tensor of shape (2 * 2) with the global angle error and joint position error. Mean and standard deviation of the errors are returned.
        """
        y_pred = matrix_to_axis_angle(y_pred) 
        y_true = matrix_to_axis_angle(y_true)
        y_pred = y_pred.to(self.device).view(-1, 24, 3)
        y_true = y_true.to(self.device).view(-1, 24, 3)
        y_pred[:, ignored_joint_set] = torch.Tensor(
            [0.0, 0.0, 0.0]).to(self.device)
        y_true[:, ignored_joint_set] = torch.Tensor(
            [0.0, 0.0, 0.0]).to(self.device)

        y_pred_joints, y_pred_vertices = self.body_model.forward_kinematics(
            y_pred)
        y_true_joints, y_true_vertices = self.body_model.forward_kinematics(
            y_true)

        offset = (y_true_joints[:, 0] - y_pred_joints[:, 0]).unsqueeze(1)

        angle_mask = torch.tensor([1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]) 
        global_angle_err = angle_between_rotations_axis_angles(
            SMPL_local_to_global(y_pred), SMPL_local_to_global(y_true))
        global_angle_err = global_angle_err[:, angle_mask]

        position_mask = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
        joint_err = (y_pred_joints + offset - y_true_joints).norm(dim=2) * 100
        joint_err = joint_err[:, position_mask] 

        return global_angle_err.mean(dim=0), joint_err.mean(dim=0),
        #return [global_angle_err.mean(dim=0), global_angle_err.std(dim=0)] , [joint_err.mean(dim=0), joint_err.std(dim=0)],


    def eval_raw(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Evaluates the prediction of SMPL sequences against ground truth in five criteria.

        B: Batch size
        S: Sequence length

        Args:
            y_pred (torch.Tensor): SMPL predicted pose tensor that can be reshaped to (B * S * 24 * 3)
            y_true (torch.Tensor): SMPL true pose tensor that can be reshaped to (B * S * 24 * 3)

        Returns:
            torch.Tensor: Returns a tensor of shape (5 * 1) with the SIP error, global angle error, joint position error, mesh error and jerk.
        """
        y_pred = matrix_to_axis_angle(y_pred) 
        y_true = matrix_to_axis_angle(y_true)
        y_pred = y_pred.to(self.device).view(-1, 24, 3)
        y_true = y_true.to(self.device).view(-1, 24, 3)
        y_pred[:, ignored_joint_set] = torch.Tensor([0.0, 0.0, 0.0]).to(self.device)
        y_true[:, ignored_joint_set] = torch.Tensor([0.0, 0.0, 0.0]).to(self.device)

        y_pred_joints, y_pred_vertices = self.body_model.forward_kinematics(
            y_pred)
        y_true_joints, y_true_vertices = self.body_model.forward_kinematics(
            y_true)

        offset = (y_true_joints[:, 0] - y_pred_joints[:, 0]).unsqueeze(1)

        global_angle_err = angle_between_rotations_axis_angles(
            SMPL_local_to_global(y_pred), SMPL_local_to_global(y_true))

        joint_mask = torch.tensor([1, 2, 16, 17])
        masked_global_angle_err = global_angle_err[:, joint_mask]

        joint_err = (y_pred_joints + offset - y_true_joints).norm(dim=2) * 100

        vertices_err = (y_pred_vertices + offset -
                        y_true_vertices).norm(dim=2) * 100

        jerk = calc_jerk(y_pred_joints).norm(dim=2) * 0.01

        distance = (y_true_joints - y_true_joints[:, 0].unsqueeze(1)).norm(dim=2) * 100

        return masked_global_angle_err, global_angle_err, joint_err, vertices_err, jerk, distance



    @staticmethod
    def print(errors, name='', mode='no_mode'):
        header=['SIP Error (deg)', 'Angular Error (deg)', 'Positional Error (cm)',
                                'Mesh Error (cm)', 'Jitter Error (100m/s^3)']
        
        for i, metric in enumerate(header):
            print('%s: %.2f (+/- %.2f)' % (metric, errors[i, 0], errors[i, 1]))

        if os.path.isdir(os.path.join(paths['experiments'], name)):
            if not os.path.isdir(os.path.join(paths['experiments'], name, 'evaluation')):
                os.mkdir(os.path.join(paths['experiments'], name, 'evaluation'))
            path = os.path.join(paths['experiments'], name, 'evaluation', 'evaluate_'+ mode +'_results.csv')
            with open((path), 'a', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(torch.t(errors).detach().cpu().numpy())
                print('Data has been stored in', path)
        else:
            print('Data has not been stored')

 

    @staticmethod
    def print_joints(angles, joints, name=''):
        angles = angles.cpu().detach().numpy()
        joints = joints.cpu().detach().numpy()

        print("{:<20}|".format("Metrices"), ("{:<10}" *15).format("L-Hip", "R-Hip", "Spine1", "L-Knee", "R-Knee", "Spine2", 
            "Spine3", "Neck", "L-Collar", "R-Collar", "Head", "L-Shoul.", "R-Shoul.", "L-Elbow", "R-Elbow"))
        print("{:<20}|".format('Angle Err (deg)'), ("{:<10.3f}" *15).format(angles[0], angles[1], angles[2], angles[3], angles[4], 
            angles[5], angles[6], angles[7], angles[8], angles[9], angles[10], angles[11], angles[12], angles[13], angles[14]))

        print("{:<20}|".format("Metrices"), ("{:<10}" *20).format("L-Knee", "R-Knee", "Spine2",  "L-Ankle", "R-Ankle",
            "Spine3", "L-Foot", "R-Foot", "Neck", "L-Collar", "R-Collar", "Head", "L-Shoul.", "R-Shoul.", "L-Elbow", "R-Elbow",
            "L-Wrist", "R-Wrist", "L-Hand", "R-Hand"))
        print("{:<20}|".format('Position Err (cm)'), ("{:<10.3f}" *20).format(joints[0], joints[1], joints[2], joints[3], joints[4], 
            joints[5], joints[6], joints[7], joints[8], joints[9], joints[10], joints[11], joints[12], joints[13], joints[14],
            joints[15], joints[16], joints[17], joints[18], joints[19]))


        angle_header = ["L-Hip", "R-Hip", "Spine-1", "L-Knee", "R-Knee", "Spine-2", 
            "Spine-3", "Neck", "L-Collar", "R-Collar", "Head", "L-Shoulder", "R-Shoulder", "L-Elbow", "R-Elbow"]
        position_header = ["L-Knee", "R-Knee", "Spine-2", "L-Ankle", "R-Ankle", "Spine-3", "L-Foot", "R-Foot",
            "Neck", "L-Collar", "R-Collar", "Head", "L-Shoulder", "R-Shoulder", "L-Elbow", "R-Elbow",
            "L-Wrist", "R-Wrist", "L-Hand", "R-Hand"]

        if os.path.isdir(os.path.join(paths['experiments'], name)):
            path = os.path.join(paths['experiments'], name, 'evaluation')
            if not os.path.isdir(path):
                os.mkdir(path)
            path_angle = os.path.join(path, 'evaluate_joints_angle_results.csv')
            with open((path_angle), 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(angle_header)
                writer.writerow(angles)
                print('Angle error has been stored in', path_angle)
            path_position = os.path.join(path, 'evaluate_joints_position_results.csv')
            with open((path_position), 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(position_header)  
                writer.writerow(joints)
                print('Angle error has been stored in', path_position)
        else:
            print('Data has not been stored')    