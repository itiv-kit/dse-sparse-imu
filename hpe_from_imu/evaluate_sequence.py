from platform import architecture
import torch
import torch.utils.tensorboard
from poutyne import set_seeds
from torchinfo import summary

from hpe_from_imu.framework import IMUExperiment, parse_train_args
from hpe_from_imu.utils import latex_utils, script_utils

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


    config_arg = script_utils.DIP_config_from_name(name)
    #experiment.evaluate_sequence(config_arg, config_arg, 10, 1, "_a")
    experiment.evaluate_sequence(config_arg, config_arg, 10, 2, "")
    #experiment.evaluate_sequence(config_arg, config_arg, 10, 3, "_a")
    #experiment.evaluate_sequence(config_arg, config_arg, 10, 4, "_a")
    #experiment.evaluate_sequence(config_arg, config_arg, 10, 5, "")

