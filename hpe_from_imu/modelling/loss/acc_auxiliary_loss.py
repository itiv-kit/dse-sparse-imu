from hpe_from_imu.configuration import Config
from hpe_from_imu.configuration import Constants as C
import torch.nn as nn

IMU_mask = Config(C.config_path)["TP_joint_set"]["reduced"]

class AccAuxiliaryLoss(nn.Module):
    def __init__(self, aux_weight=0.2):
        super().__init__()
        self.aux_weight = aux_weight
        self.mse_loss = nn.MSELoss()
        
    def forward(self, y_pred, y_true):
        n_acc = len(y_pred[0,0])-(len(IMU_mask)*9)
        loss_normal = self.mse_loss(y_pred[:, :, :-n_acc], y_true[:, :, :-n_acc])
        loss_aux = self.mse_loss(y_pred[:, :, -n_acc:], y_true[:, :, -n_acc:])
        loss = loss_normal + self.aux_weight * loss_aux
        return loss

    def __str__(self):
        return "AccAuxiliaryLoss"
