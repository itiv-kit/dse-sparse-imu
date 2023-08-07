import numpy as np
import torch
import os
import csv

from hpe_from_imu.configuration import Config
from hpe_from_imu.configuration import Constants as C


conf = Config(C.config_path)
paths = conf["paths"]

class LatencyEvaluator():
    """
    Evaluates the inference latency for a given network.
    The measured latency only includes the actual inference.
    Memory allocation or data transfers are not included in the timing.
    Includes GPU warm-up before performing the test and takes asynchronous execution time of the GPU into account.
    Based on https://towardsdatascience.com/the-correct-way-to-measure-inference-time-of-deep-neural-networks-304a54e5187f
    """

    def __init__(self, network: torch.nn.Module,  reps=1000, warm_up_reps=1000):
        """
        Args:
        network (torch.nn.Module): The network that should be tested.       
        reps (int, optional): The number of repetitions for latency test before averaging the results. Defaults to 1000.
        warm_up_reps (int, optional): The number of warm-up repetitions before the evalutation starts. Necessary because gpu takes time to initalize. Defaults to 1000.
        """
        self._network = network.to(torch.device("cuda"))
        self._reps = reps
        self._warm_up_reps = warm_up_reps

    def eval(self, input: torch.Tensor):
        """
        Evaluates the inference latency for a given network.

        Args:
        input (torch.Tensor): The required input size for the given network.

        Returns:
        dict[str, Any]: A dictionary containing the results (mean and standard deviation) and accompanying information (warm-up repetitions, repetitions and input shape)
        """
        starter, ender = torch.cuda.Event(
            enable_timing=True), torch.cuda.Event(enable_timing=True)
        timings = np.zeros((self._reps, 1))

        # GPU-WARM-UP
        for _ in range(self._warm_up_reps):
            _ = self._network(input)

        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(self._reps):
                starter.record()
                _ = self._network(input)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        mean = np.average(timings)
        std = np.std(timings)
        return {
            "shape": input.shape,
            "warm_up_reps": self._warm_up_reps,
            "reps": self._reps,
            "mean": mean,
            "std": std
        }
    
    def write_csv(self, data, name):
        if os.path.isdir(os.path.join(paths['workspace_dir'], 'experiments', name)):
            path = os.path.join(paths['workspace_dir'], 'experiments', name, 'evaluation')
            if not os.path.isdir(path):
                os.mkdir(path)
            path = os.path.join(path, 'evaluate_latency.csv')
        
        data['shape'] = str(np.array(data['shape']))
        
        if os.path.isfile(path):
            header = [] 
            with open(path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    header.append(row)
                    break
            with open(path, "a", encoding='UTF8', newline='') as f:
                #writer = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
                writer = csv.writer(f)
                writer.writerow(data.values())
        else:
            with open((path), 'w', encoding='UTF8', newline='') as f:
                w = csv.DictWriter(f, data.keys(), extrasaction='ignore')
                w.writeheader()
                w.writerow(data)

    @staticmethod
    def print(input):
        print(
            "shape: {shape}, warm-up reps: {warm_up_reps}, reps: {reps}, mean (ms): {mean}, std (ms): {std}".format(**input))