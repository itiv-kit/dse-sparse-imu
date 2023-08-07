import csv
import os
import threading
import numpy as np
from datetime import datetime
from time import sleep

import torch
from pynvml import *
import matplotlib.pyplot as plt

from hpe_from_imu.configuration import Config
from hpe_from_imu.configuration import Constants as C

conf = Config(C.config_path)
paths = conf["paths"]

class PowerDrawEvaluator():
    def __init__(self, gpu_id: int, network: torch.nn.Module, path: str = "", name: str = "", reps=10000, batch_size :int = 0):
        self.gpu_id = gpu_id
        self.network = network.to(torch.device("cuda"))
        self.path = path
        self.name = name
        self.reps = reps
        self.batch_size = batch_size

    def eval(self, input: torch.Tensor):
        power_measure_thread = PowerMeasureThread(self.gpu_id, self.path, self.name, self.batch_size)
        power_measure_thread.start()
        sleep(20)
        with torch.no_grad():
            for _ in range(self.reps):
                _ = self.network(input)
        power_measure_thread.measure = False

class PowerMeasureThread(threading.Thread):
    """
    Class to spawn thread running concurrently to forward pass to measure GPU power consumption.
    Uses the NVIDIA management library to measure power draw. Saves data to csv file.
    """

    def __init__(self, gpu_id: int, path: str, name: str, batch_size):
        threading.Thread.__init__(self)
        self.path = path
        self.name = name
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        self.measure = True

    def write_csv(self, data):
        if os.path.isdir(os.path.join(paths['workspace_dir'], 'experiments', self.name)):
            path = os.path.join(paths['workspace_dir'], 'experiments', self.name, 'evaluation')
            if not os.path.isdir(path):
                os.mkdir(path)
            path = os.path.join(path, 'evaluate_power_draw.csv')
        
        if os.path.isfile(path):
            header = [] 
            with open(path, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    header.append(row)
                    break
            with open(path, "a", encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data.values())
        else:
            with open((path), 'w', encoding='UTF8', newline='') as f:
                w = csv.DictWriter(f, data.keys(), extrasaction='ignore')
                w.writeheader()
                w.writerow(data)

    def plot_measurement(self, data, desc='power_draw'):

        if os.path.isdir(os.path.join(paths['workspace_dir'], 'experiments', self.name)):
            path = os.path.join(paths['workspace_dir'], 'experiments', self.name, 'evaluation')
            if not os.path.isdir(path):
                os.mkdir(path)
            path = os.path.join(path, 'plot_'+desc+'.png')

        fig = plt.figure()
        plot_data = np.asarray(data)
        mu = plot_data.mean()
        x_max = plot_data.argmax()
        x_min = plot_data.argmin()
        plt.axhline(y=mu, color='g', label=("Mean = "+str(mu.round(2))))
        plt.scatter(x=x_max, y=plot_data[x_max], color='r', label=("Max = {:.2f} at {:.0f}".format(plot_data.max(), x_max)))
        plt.scatter(x=x_min, y=plot_data[x_min], color='r', label=("Min = {:.2f} at {:.0f}".format(plot_data.min(), x_min)))
        plt.legend(bbox_to_anchor=(1,1), loc='upper left')
        plt.plot(plot_data, ls= '-')
        plt.grid()
        plt.xlabel("n-Measurements")
        plt.ylabel(("Power Consumption in W" if desc=='power_draw' else "Utilization in %"))
        plt.title(desc)
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)
    
    def run(self):
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        print(f"GPU Count: {device_count}")
        handle_gpu_0 = nvmlDeviceGetHandleByIndex(0)
        handle_gpu_1 = nvmlDeviceGetHandleByIndex(1)
        device_name_gpu_0 = nvmlDeviceGetName(handle_gpu_0)
        device_name_gpu_1 = nvmlDeviceGetName(handle_gpu_1)
        print(f"Start of Measurement: {datetime.now()}")
        measurements_gpu_0, measurements_gpu_1 = [], []
        while(self.measure == True):
            use_gpu_0 = nvmlDeviceGetUtilizationRates(handle_gpu_0)
            measurements_gpu_0.append(
                [datetime.now(), device_name_gpu_0, nvmlDeviceGetPowerUsage(handle_gpu_0)/1000, use_gpu_0.gpu, use_gpu_0.memory])
            use_gpu_1 = nvmlDeviceGetUtilizationRates(handle_gpu_1)
            measurements_gpu_1.append(
                [datetime.now(), device_name_gpu_1, nvmlDeviceGetPowerUsage(handle_gpu_1)/1000, use_gpu_1.gpu, use_gpu_1.memory])
            sleep(0.02)
        print(f"End of Measurement: {datetime.now()}")
        nvmlShutdown()
        if np.mean(use_gpu_0.gpu) >= np.mean(use_gpu_1.gpu):
            device_name = device_name_gpu_0
            measurements = measurements_gpu_0
            print(f"GPU ID: {0}")
            print(f"Testing Device: {device_name}")
        else:
            device_name = device_name_gpu_1
            measurements = measurements_gpu_1
            print(f"GPU ID: {1}")
            print(f"Testing Device: {device_name}")

        writer = csv.writer(open(self.path, "a"))
        writer.writerows(measurements)
        power_draw, gpu_use, memory_use = [],[],[] 
        for _, _, power, gpu, memory in measurements:
            power_draw.append(power)
            gpu_use.append(gpu)
            memory_use.append(memory)
        power_draw_mean = np.mean(power_draw)
        power_draw_min = min(power_draw)
        power_draw_max = max(power_draw)

        print(
            f"Min Power Draw: {power_draw_min}W, Max Power Draw: {power_draw_max}W")
        
        data = {
            'Device' : device_name,
            'GPU Count' : device_count,
            'Batch Size' : self.batch_size,
            'Mean Power Draw' : power_draw_mean.round(3),
            'Min Power Draw' : power_draw_min,
            'Max Power Draw' : power_draw_max,
            'Unit' : 'Watt'
        } 
        self.write_csv(data)
        self.plot_measurement(power_draw)
        self.plot_measurement(gpu_use, desc='gpu_use')
        self.plot_measurement(memory_use, desc='memory_use')