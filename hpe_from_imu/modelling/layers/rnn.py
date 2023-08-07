import torch.nn
from hpe_from_imu.configuration import Config
from hpe_from_imu.configuration import Constants as C
from torch.nn.functional import relu

conf = Config(C.config_path)
paths = conf["paths"]
tp_joint_set = conf["TP_joint_set"]


class MyRNN(torch.nn.Module):
    """
    An RNN Module including a linear input layer, an RNN, and a linear output layer.

    Based on https://github.com/Xinyu-Yi/TransPose/blob/d37d617bbee044e5c1ad2e853f883b1001a5f87b/net.py#L7
    """

    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=True, dropout=0.2):
        super(MyRNN, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.linear1 = torch.nn.Linear(n_input, n_hidden)
        self.rnn = torch.nn.LSTM(
            n_hidden, n_hidden, n_rnn_layer, bidirectional=bidirectional, batch_first=True)
        self.linear2 = torch.nn.Linear(
            n_hidden * (2 if bidirectional else 1), n_output)

    def forward(self, x, h=None):
        x, h = self.rnn(relu(self.linear1(self.dropout(x))), h)
        return self.linear2(x)


class PoseNet(torch.nn.Module):
    """
    A model for pose estimation from IMU inputs based on the TransPose model but without translation estimation.

    Based on https://github.com/Xinyu-Yi/TransPose/blob/d37d617bbee044e5c1ad2e853f883b1001a5f87b/net.py#L23
    """

    def __init__(self):
        super().__init__()
        n_imu = 6 * 3 + 6 * 9
        self.pose_s1 = MyRNN(n_imu, tp_joint_set["n_leaf"] * 3, 256)
        self.pose_s2 = MyRNN(tp_joint_set["n_leaf"] * 3 + n_imu,
                             tp_joint_set["n_full"] * 3, 64)
        self.pose_s3 = MyRNN(tp_joint_set["n_full"] * 3 + n_imu,
                             tp_joint_set["n_reduced"] * 3, 128)

    def forward(self, imu):
        leaf_joint_position = self.pose_s1.forward(imu)
        full_joint_position = self.pose_s2.forward(
            torch.cat((leaf_joint_position, imu), dim=2))
        global_reduced_pose = self.pose_s3.forward(
            torch.cat((full_joint_position, imu), dim=2))
        return global_reduced_pose
