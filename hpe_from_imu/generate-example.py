import os
import glob
import torch
from poutyne import set_seeds
from hpe_from_imu.configuration import Config
from hpe_from_imu.configuration import Constants as C
from hpe_from_imu.dataloader import getExampleDataLoader
from hpe_from_imu.preprocess import prep_DIP_example
from hpe_from_imu.framework import IMUExperiment, parse_train_args
from hpe_from_imu.modelling import BaseIMUNet
from hpe_from_imu.utils import script_utils

conf = Config(C.config_path)
paths = conf["paths"]
dataset_paths = conf["dataset_paths"]
reduced_joint_set = conf["TP_joint_set"]["reduced"]

def custom_example(raw_path_key, sub, mot, spec):
    # subject, motion, specifier e.g. 's_10-01a' is 10, 1, '_a' or 's_10_02' is 10, 2, ''

    print("Loading customized DIP-IMU dataset")
    acc = torch.load(os.path.join(paths[raw_path_key], "vaccs.pt"))
    ori = torch.load(os.path.join(paths[raw_path_key], "vrots.pt"))
    pose = torch.load(os.path.join(paths[raw_path_key], "poses.pt"))

    os.makedirs(paths["example_dir"], exist_ok=True)
    file_path = os.path.join(paths["DIP_IMU"], "s_{:0=2d}".format(sub), "{:0=2d}{}.pkl".format(mot, spec))
    files = glob.glob(os.path.join(paths["DIP_IMU"], '*/*.pkl'))
    files.sort()
    idx = files.index(file_path)
                
    file = "s_{:0=2d}-{:0=2d}{}".format(sub, mot, spec)
    example_path = os.path.join(paths["example_dir"], file + "_synth")
    torch.save({'acc': acc[idx:(idx+1)], 'ori': ori[idx:(idx+1)], 'pose': pose[idx:(idx+1)]},
               (example_path + "-example.pt"))
    print('Created custom example dataset and saved it at',
          paths["example_dir"])

    return example_path

def generate_example_batch(name, trainer, online=0):
    # creates examples for five sequences from subject 10
    for i in range(5):

        sub = 10
        mot = i+1
        if (mot == 2 or mot == 5):
            spec = ''
        else:
            spec = '_a'

        config_arg = script_utils.DIP_config_from_name(name)
        DIP_example_path = custom_example(config_arg, sub, mot, spec) # Comment out if using a unsynthesized dataset

        DIP_example = getExampleDataLoader(
            DIP_example_path, model.transforms_in["DIP"])

        pred_off, gt = trainer.predict_offline(DIP_example)
        torch.save(pred_off, os.path.join(
            paths["example_dir"], DIP_example_path + "-" + trainer.simplified_name + "-offline-pose.pt"))
        torch.save(gt, os.path.join(
            paths["example_dir"], DIP_example_path + "-gt-pose.pt"))

        if online:
            pred_on, gt = trainer.predict_online(DIP_example)
            torch.save(pred_on, os.path.join(
            paths["example_dir"], DIP_example_path + "-" + trainer.simplified_name + "-online-pose.pt"))

        print("Saved predictions under",
            DIP_example_path + "-" + trainer.simplified_name)
    print("\nFinished to create examples\n")


if __name__ == '__main__':
    set_seeds(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, hyper_parameters, datasets, name, cfg = parse_train_args()
    model: BaseIMUNet = model.to(device)

    trainer = IMUExperiment(model, hyper_parameters,
                                datasets=datasets, device=device, name=name)

    # subject, motion, specifier e.g. 's_10-01a' is 10, 1, '_a' or 's_10_02' is 10, 2, ''
    sub, mot, spec = 10, 1, '_a'
    #sub, mot, spec = 10, 2, ''
    #sub, mot, spec = 10, 3, '_a'
    #sub, mot, spec = 10, 4, '_a'
    #sub, mot, spec = 10, 5, ''
    
    # use if preparing real DIP-IMU example
    DIP_example_path = "s_{:0=2d}-{:0=2d}{}".format(sub, mot, spec)
    prep_DIP_example("s_{:0=2d}".format(sub), "{:0=2d}{}".format(mot, spec), mask="DIP_IMU_mask")
    
    # use if preparing synthetic DIP example
    #config_arg = script_utils.DIP_config_from_name(name)
    #DIP_example_path = custom_example(config_arg, sub, mot, spec)

    DIP_example = getExampleDataLoader(
        DIP_example_path, model.transforms_in["DIP"])

    #pred_on, gt = trainer.predict_online(DIP_example)
    pred_off, gt = trainer.predict_offline(DIP_example)

    torch.save(pred_off, os.path.join(
        paths["example_dir"], DIP_example_path + "-" + trainer.simplified_name + "-offline-pose.pt"))
    #torch.save(pred_on, os.path.join(
    #    paths["example_dir"], DIP_example_path + "-" + trainer.simplified_name + "-online-pose.pt"))
    torch.save(gt, os.path.join(
        paths["example_dir"], DIP_example_path + "-gt-pose.pt"))
    print("Saved predictions under",
        DIP_example_path + "-" + trainer.simplified_name)
