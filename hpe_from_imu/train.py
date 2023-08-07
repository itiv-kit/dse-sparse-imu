from platform import architecture
import torch
import torch.utils.tensorboard
from poutyne import set_seeds
from torchinfo import summary

from hpe_from_imu.framework import IMUExperiment, parse_train_args
from hpe_from_imu.utils import script_utils

if __name__ == '__main__':
    set_seeds(42)

    model, hyper_parameters, datasets, name, config = parse_train_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    #(1, n_input) dynamically from config
    input_layer = list(config["architecture"][0].keys())
    n_input = config["architecture"][0][input_layer[0]]["n_input"]
    summary(model, (1, int(n_input)), verbose=2, col_names=("input_size",
                                                  "output_size", "num_params", "kernel_size", "mult_adds"))

    experiment = IMUExperiment(model, hyper_parameters,
                                datasets=datasets, device=device, name=name, config=config)
    #experiment.evaluate_latency((26, int(n_input)))
    #experiment.evaluate_power_draw((26, int(n_input)))
    experiment.train(config["parameters"]["train_epochs"])
    experiment.evaluate_offline(datasets['test_data'])
    # experiment.evaluate_online("TC")
    experiment.finetune(config["parameters"]["train_epochs"] + config["parameters"]["finetune_epochs"])
    experiment.evaluate_offline(datasets['test_data'])
    #experiment.evaluate_joints("DIP_spine3_arm_test")
    # experiment.evaluate_online("TC")
    #experiment.finetune(100)
    #experiment.evaluate_offline(datasets['test_data'])
    #experiment.evaluate_joints(datasets['test_data'])
    #experiment.evaluate_online(datasets['test_data'])
