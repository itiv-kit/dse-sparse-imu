import csv
import os 
import glob
import numpy as np

import matplotlib.pyplot as plt
from hpe_from_imu.configuration import Config
from hpe_from_imu.configuration import Constants as C

conf = Config(C.config_path)
paths = conf["paths"]

def DIP_dataset_from_name(name:str):
    config_name = reduced_name_from_name(name)
    train_dataset = "DIP_"+config_name+"_train"
    test_dataset = "DIP_"+config_name+"_test"
    validation_dataset = "DIP_"+config_name+"_valdation"
    return train_dataset, test_dataset, validation_dataset

def DIP_config_from_name(name:str):
    config_name = reduced_name_from_name(name)
    config_arg ="DIP_"+config_name
    return config_arg

def reduced_name_from_name(name:str):
    cut_out = ["DIPNet_", "05noise-", "AMASS_", "-Adam", "-AccAuxiliaryLoss"]
    # removes DATE-TIME
    config_name = name[14:]
    # Add further parts that should be cut out of name-string
    for i, part in enumerate(cut_out):
        config_name = config_name.replace(part,"")
    return config_name
