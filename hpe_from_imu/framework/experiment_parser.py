import argparse
import copy
from os import listdir
from os.path import join as join_path

import hpe_from_imu.dataloader as dataloader
import hpe_from_imu.modelling as modelling
import poutyne
import torch
import torch.nn
import yaml
from hpe_from_imu.configuration import Config
from hpe_from_imu.configuration import Constants as C
from hpe_from_imu.framework.build import (Builder, build_network,
                                          build_parameters, build_transforms)
from hpe_from_imu.modelling.base.imu_net import BaseIMUNet

conf = Config(C.config_path)
paths = conf["paths"]
conf_dataset_paths = conf["dataset_paths"]


def _setup_train_args() -> argparse.ArgumentParser:
    """Setups all the arguments and parses them.

    Returns:
        argparse.ArgumentParser: The parsed argument parser for this script.
    """
    p = argparse.ArgumentParser(
        description='Train models implmenting BaseIMUNet.')

    p.add_argument('-c', '--config',
                   dest='config',
                   help='Config yaml file',
                   type=str,
                   choices=listdir(paths["experiments_config"]),
                   default=None)

    p.add_argument('-n', '--name',
                   dest='name',
                   help='Name of the experiment that should be recovered',
                   type=str,
                   choices=listdir(paths["experiments"]),
                   default=None)

    return p.parse_args()


def _parse_datasets(cfg):
    datasets = {
        "train_data": cfg["dataset"]["train"],
        "validation_data": cfg["dataset"]["validation"],
        "fine_tune_data": cfg["dataset"]["finetune"],
        "test_data": cfg["dataset"]["test"],
        "use_general_dataset": cfg["dataset"]["use_general_dataset"],
    }

    return datasets


def _parse_model(cfg, builder):
    layers = build_network(architecture=cfg["architecture"], builder=builder)
    if cfg["dataset"]["use_general_dataset"]:
        return BaseIMUNet(cfg["name"], layers, {}, [])
    transforms_in, transforms_out = build_transforms(
        transforms=cfg["dataset"]["transforms"], builder=builder)
    return BaseIMUNet(cfg["name"], layers, transforms_in, transforms_out)


def _load_cfg(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)


def _merge_cfg(cfg_a, cfg_b):
    cfg = cfg_a
    for k, v in cfg_b.items():
        cfg[k] = cfg[k] | cfg_b[k] if type(v) is dict else v
    return cfg


def _parse_config(config):
    cfg_default = _load_cfg(paths["experiments_base_config"])
    cfg = _load_cfg(config)
    cfg_merged = _merge_cfg(cfg_default, cfg)
    return cfg_merged


def parse_train_args():
    cfg_path, name = _parse_args()
    cfg = _parse_config(cfg_path)
    cfg_orig = copy.deepcopy(cfg)
    builder = Builder(torch.nn.__dict__, modelling.__dict__,
                      dataloader.__dict__, poutyne.__dict__)
    datasets = _parse_datasets(cfg)
    net = _parse_model(cfg, builder=builder)
    params = build_parameters(cfg["parameters"], builder=builder)
    return net, params, datasets, name, cfg_orig


def _parse_args():
    args = _setup_train_args()
    assert args.config is not None or args.name is not None, "Pls either specify a config or an existing experiment"
    if args.name is not None:
        print("Loading existing config")
        cfg_path = join_path(paths["experiments"], args.name, "config.yml")
    elif args.config is not None:
        print("Creating new config")
        cfg_path = join_path(paths["experiments_config"], args.config)
    return cfg_path, args.name
