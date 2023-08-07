from collections import OrderedDict

import torch
from poutyne import Model, set_seeds

from hpe_from_imu.configuration import Config
from hpe_from_imu.configuration import Constants as C
from hpe_from_imu.modelling import BaseIMUNet, MyRNN, PoseNet

conf = Config(C.config_path)
paths = conf["paths"]
dataset_paths = conf["dataset_paths"]
reduced_joint_set = conf["TP_joint_set"]["reduced"]


def replace_in_key(dict: OrderedDict, toReplace: str, replaceWith: str):
    new_keys = [key.replace(toReplace, replaceWith)
                for key in dict.keys()]
    return OrderedDict(zip(new_keys, list(dict.values())))


if __name__ == '__main__':
    set_seeds(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net_S1 = BaseIMUNet("S1",  torch.nn.Sequential(MyRNN(n_input=72, n_output=15,
                        n_hidden=256)), {}, []).to(device)
    net_S2 = BaseIMUNet("S2", torch.nn.Sequential(MyRNN(n_input=87, n_output=69,
                        n_hidden=64)), {}, []).to(device)
    net_S3 = BaseIMUNet("S3", torch.nn.Sequential(MyRNN(n_input=141, n_output=45,
                        n_hidden=128)), {}, []).to(device)

    model_S1 = Model(net_S1, optimizer=None, loss_function=None).to(
        device)
    model_S1.load_weights("experiments/posenet/s1_weights")
    model_S2 = Model(net_S2, optimizer=None, loss_function=None).to(
        device)
    model_S2.load_weights("experiments/posenet/s2_weights")
    model_S3 = Model(net_S3, optimizer=None, loss_function=None).to(
        device)
    model_S3.load_weights("experiments/posenet/s3_weights")

    weights_s1 = replace_in_key(
        model_S1.get_weight_copies(), ".0.", ".0.pose_s1.")
    weights_s2 = replace_in_key(
        model_S2.get_weight_copies(), ".0.", ".0.pose_s2.")
    weights_s3 = replace_in_key(
        model_S3.get_weight_copies(), ".0.", ".0.pose_s3.")

    weights_total = weights_s1 | weights_s2 | weights_s3
    torch.save(weights_total, "experiments/posenet/total_weights")

    net_total = BaseIMUNet("Test", layers=torch.nn.Sequential(PoseNet()))
    model_total = Model(net_total, optimizer=None,
                        loss_function=None).to(device)
    model_total.load_weights("experiments/posenet/total_weights")
