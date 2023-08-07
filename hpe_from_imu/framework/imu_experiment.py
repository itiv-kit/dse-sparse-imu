import os
import shutil
from datetime import datetime
from hpe_from_imu.preprocess import preprocess_DIP_TP_custom_synth, preprocess_DIP_TP_from_temp

import torch
import numpy as np
import yaml
import csv
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import PercentFormatter
from hpe_from_imu.configuration import Config
from hpe_from_imu.configuration import Constants as C
from hpe_from_imu.preprocessing import  custom_synthesize_DIP, custom_synthesize_AMASS, remove_temp_data
from hpe_from_imu.dataloader import getGeneralDataLoader, getIMUDataLoader
from hpe_from_imu.evaluation import (LatencyEvaluator, PoseEvaluator,
                                     PowerDrawEvaluator)
from hpe_from_imu.modelling import BaseIMUNet
from hpe_from_imu.utils import unfold_to_sliding_windows
from poutyne import ClipNorm, EarlyStopping, ModelBundle, TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

conf = Config(C.config_path)
paths = conf["paths"]
conf_dataset_paths = conf["dataset_paths"]
conf_dataset_base = conf["dataset_base"]
reduced_joint_set = conf["TP_joint_set"]["reduced"]


class IMUExperiment():

    def __init__(self, model: BaseIMUNet, hyper_parameters, device, datasets: dict[str, str], name: str = None, config: dict = None):
        super().__init__()
        self._model = model
        self._device = device

        self._parse_hyper_parameters(hyper_parameters)
        self._parse_data(datasets)
        self._parse_name(name)

        self._model_bundle = self._build_model_bundle()
        if name is None:
            self._save_config(config)

    def _save_config(self, config):
        os.makedirs(paths["experiments"] + self._name, exist_ok=True)
        with open(paths["experiments"] + self._name + "/config.yml", 'w') as yaml_output:
            yaml.safe_dump(config, yaml_output)

    def _parse_name(self, name):
        if name is None:
            self._name = self._build_name()
        else:
            self._name = name

    def _parse_hyper_parameters(self, hyper_parameters):
        self._epochs = hyper_parameters["train_epochs"]
        self._optimizer = hyper_parameters["optimizer"]
        self._loss_function = hyper_parameters["loss_function"][0] if type(
            hyper_parameters["loss_function"]) is list else hyper_parameters["loss_function"]
        self._batch_metrics = hyper_parameters["batch_metrics"]
        self._epoch_metrics = hyper_parameters["epoch_metrics"]
        self._monitor_metric = hyper_parameters["monitor_metric"]
        self._lr_schedulers = hyper_parameters["lr_schedulers"]
        self._batch_size = hyper_parameters["batch_size"]
        self._num_past_frames = hyper_parameters["num_past_frames"]
        self._num_future_frames = hyper_parameters["num_future_frames"]
        self._total_frames = self._num_past_frames + self._num_future_frames + 1

    def _parse_data(self, datasets):
        self._train_data_str = datasets["train_data"]
        self._use_general_dataset = datasets["use_general_dataset"]
        self._train_data = self._get_data_loader(datasets["train_data"])
        self._validation_data = self._get_data_loader(
            datasets["validation_data"])
        self._fine_tune_data = self._get_data_loader(
            datasets["fine_tune_data"])
        self._test_data_str = datasets["test_data"]
        self._test_data = self._get_data_loader(datasets["test_data"])

    def _get_data_loader(self, dataset: str):
        if dataset is None:
            return None
        path = conf_dataset_paths[dataset]
        dataset_base = conf_dataset_base[dataset]
        if self._use_general_dataset:
            return getGeneralDataLoader(path, batch_size=self._batch_size)
        return getIMUDataLoader(path, batch_size=self._batch_size, transforms=self._get_inwards_transforms(dataset_base))

    def _get_inwards_transforms(self, dataset: str):
        return self._model.transforms_in[dataset]

    def _apply_outwards_transforms(self, y):
        if self._model.transforms_out is not None:
            for transformation in self._model.transforms_out:
                y = transformation(y)
        return y

    def _build_name(self):
        name = datetime.now().strftime("%Y%m%d_%H%M%S")
        name += "-" + str(self._model)
        name += "-" + self._train_data_str
        name += "-" + self._optimizer["optim"]
        name += "-" + str(self._loss_function)
        return name

    def _build_model_bundle(self):
        return ModelBundle.from_network(
            paths["experiments"] + self._name,
            self._model,
            optimizer=self._optimizer,
            loss_function=self._loss_function,
            batch_metrics=self._batch_metrics,
            epoch_metrics=self._epoch_metrics,
            monitor_metric=self._monitor_metric,
            device=self._device
        )

    def train(self, epochs):
        assert self._train_data is not None, "No training data"
        if self._validation_data is not None:
            callbacks = [EarlyStopping(
                monitor="val_loss", min_delta=0, patience=5, verbose=True, mode="min"), ClipNorm(self._model.parameters(), 1)]
            self._model_bundle.train(
                self._train_data, self._validation_data, callbacks=callbacks, lr_schedulers=self._lr_schedulers, epochs=epochs, batches_per_step=1, keep_only_last_best=True)
        else:
            callbacks = [EarlyStopping(
                monitor="loss", min_delta=0, patience=5, verbose=True, mode="min"),  ClipNorm(self._model.parameters(), 1)]
            self._model_bundle.train(
                self._train_data, callbacks=callbacks, lr_schedulers=self._lr_schedulers, epochs=epochs, batches_per_step=1, keep_only_last_best=True)

    def finetune(self, epochs):
        assert self._fine_tune_data is not None, "No finetune data"
        print("Starting fine tuning")
        if self._validation_data is not None:
            callbacks = [EarlyStopping(
                monitor="val_loss", min_delta=0, patience=5, verbose=True, mode="min"), ClipNorm(self._model.parameters(), 1)]
            self._model_bundle.train(
                self._fine_tune_data, self._validation_data, callbacks=callbacks, lr_schedulers=self._lr_schedulers, epochs=epochs, batches_per_step=1, keep_only_last_best=True,  save_every_epoch=False)
        else:
            callbacks = [EarlyStopping(
                monitor="loss", min_delta=0, patience=5, verbose=True, mode="min"),  ClipNorm(self._model.parameters(), 1)]
            self._model_bundle.train(
                self._fine_tune_data, callbacks=callbacks, lr_schedulers=self._lr_schedulers, epochs=epochs, batches_per_step=1, keep_only_last_best=True, save_every_epoch=False)

    def test(self):
        assert self._test_data is not None, "No test data"
        self._model_bundle.test(self._test_data)

    def predict_offline(self, prediction_data):
        y_, y = self._model_bundle.infer(
            prediction_data, has_ground_truth=True,
            return_ground_truth=True, checkpoint="last")
        return self._apply_outwards_transforms(torch.from_numpy(y_)), self._apply_outwards_transforms(torch.from_numpy(y))

    def predict_online(self, prediction_data):
        y_, y = [], []
        for batch in prediction_data:
            input = batch[0]
            batch_size = input.size(0)
            feature_size = input.size(2)
            y.append(batch[1])
            windows = self._prep_sliding_windows(input)
            pred = self._model_bundle.infer_data(
                windows.reshape(batch_size, -1, feature_size), convert_to_numpy=False)
            y_.append(pred[:, self._num_past_frames::self._total_frames])
        return self._apply_outwards_transforms(torch.cat(y_)), self._apply_outwards_transforms(torch.cat(y))

    def evaluate_offline(self, dataset: str):
        print("Starting offline evaluation on", dataset)
        data = self._get_data_loader(dataset)
        evaluator = PoseEvaluator()
        errs = []
        y_pred, y_gt = self.predict_offline(data)
        for y_, y in tqdm(list(zip(y_pred, y_gt))):
            errs.append(evaluator.eval(y_, y))
        evaluator.print(torch.stack(errs).mean(dim=0), name=self._name, mode='offline')

    def evaluate_online(self, dataset: str):
        print("Starting online evaluation on", dataset)
        data = self._get_data_loader(dataset)
        evaluator = PoseEvaluator()
        errs = []
        y_pred, y_gt = self.predict_online(data)
        for y_, y in tqdm(list(zip(y_pred, y_gt))):
            errs.append(evaluator.eval(y_, y))
        evaluator.print(torch.stack(errs).mean(dim=0), name=self._name, mode='online')
        

    def evaluate_latency(self, shape, reps=1000, warm_up_reps=1000):
        network = self._model_bundle.model.network
        evaluator = LatencyEvaluator(network, reps, warm_up_reps)
        for batch_size in [1, 16, 32, 64]:
            input = torch.randn(batch_size, *shape,
                                dtype=torch.float).to(torch.device("cuda"))
            result = evaluator.eval(input)
            evaluator.print(result)

    def evaluate_power_draw(self, shape, reps=10000):
        network = self._model_bundle.model.network
        evaluator = PowerDrawEvaluator(0, network, "", reps)
        evaluator.name = self._name
        for batch_size in [1, 16, 32, 64]:
            path = os.path.join(
                paths["experiments"], self._name, f"power_draw_{batch_size}.csv")
            evaluator.path = path
            evaluator.batch_size = batch_size
            input = torch.randn(batch_size, *shape,
                                dtype=torch.float).to(torch.device("cuda"))
            evaluator.eval(input)

    def evaluate_joints(self, dataset: str):
        print("Starting offline segment evaluation on", dataset)
        data = self._get_data_loader(dataset)
        evaluator = PoseEvaluator()
        angle_errs = []
        joint_errs = []
        y_pred, y_gt = self.predict_offline(data)
        for y_, y in tqdm(list(zip(y_pred, y_gt))):
            angle, joint = evaluator.eval_joints(y_, y)
            angle_errs.append(angle)
            joint_errs.append(joint)
        evaluator.print_joints(angles=torch.stack(angle_errs).mean(dim=0), joints=torch.stack(joint_errs).mean(dim=0), name=self._name)

    def evaluate_raw(self, dataset: str):
        print("Starting offline raw data evaluation on", dataset)
        data = self._get_data_loader(dataset)
        evaluator = PoseEvaluator()
        sip_errs, angle_errs, joint_errs, vertices_errs, jerk_errs, distances = [],[],[],[],[],[]     
        y_pred, y_gt = self.predict_offline(data)
        for y_, y in tqdm(list(zip(y_pred, y_gt))):
            sip, angle, joint, vertices, jerk, distance = evaluator.eval_raw(y_, y)
            sip_errs.append(sip)
            angle_errs.append(angle)
            joint_errs.append(joint)
            vertices_errs.append(vertices)
            distances.append(distance)

        # Save error by distance
        self.plot_error_distance(angle_errs, distances, 'Angle')
        self.plot_error_distance(joint_errs, distances, 'Joint')

        # Save boxplots in evaluation-folder of experiment
        self.plot_boxplot(sip_errs,'SIP', write_csv=True)
        self.plot_boxplot(angle_errs,'Angle', fliers=True)
        self.plot_boxplot(joint_errs,'Joint')

        # Save histograms in evaluation-folder of experiment
        self.plot_histogram(sip_errs,'SIP', std=True)
        self.plot_histogram(angle_errs,'Angle', count=True)
        self.plot_histogram(joint_errs,'Joint', )
        self.plot_histogram(vertices_errs,'Vertex')



    def evaluate_sequence(self, vertex_key, joint_key, sub, mot, spec, name='temp', calc_move=False,):

        # choose sequence (interactively)
            # DIP
        sequence = "s_{:0=2d}{:0=2d}{}".format(sub, mot, spec)
        # synthesize sequence & store data at temp folder
        custom_synthesize_DIP("dummy", vertex_key, joint_key, calc_move=calc_move, sub=sub, mot=mot,spec=spec)
        # preprocess
        preprocess_DIP_TP_from_temp(sequence)
        
        self.evaluate_sequence_detailed_from_temp(sequence)

        remove_temp_data(sequence)

    def evaluate_sequence_detailed_from_temp(self, sequence, store_csv=False):
        print("Starting offline evaluation on sequence:", sequence)
        path = os.path.join(paths["workspace_dir"], "temp", sequence, "300-train.pt")
        dataset_base = "DIP" # or AMASS_TP
        data = getIMUDataLoader(path, batch_size=self._batch_size, transforms=self._get_inwards_transforms(dataset_base))
        evaluator = PoseEvaluator()

        sip, angle, joint, vertices, jerk, distance = [],[],[],[],[],[]
        y_pred, y_gt = self.predict_offline(data)
        for y_, y in tqdm(list(zip(y_pred, y_gt))):
            sip_temp, angle_temp, joint_temp, vertices_temp, jerk_temp, distance_temp  = evaluator.eval_raw(y_, y)
            sip.append(sip_temp)
            angle.append(angle_temp)
            joint.append(joint_temp)
            vertices.append(vertices_temp)
            jerk.append(jerk_temp)
            distance.append(distance_temp)

        # plot boxplots
        self.plot_boxplot(angle, 'Angle', sequence=sequence, fliers=True)
        self.plot_boxplot(joint, 'Joint', sequence=sequence, fliers=True)

        # plot histogram for Angle Error, Joint Error and Mesh Error over sequence for every single joint in histogramms/joints
        self.plot_histogram(sip, 'SIP', sequence=sequence, std=True)
        self.plot_histogram(angle, 'Angle', sequence=sequence, std=True)
        self.plot_histogram(joint, 'Joint', sequence=sequence, std=True)
        self.plot_histogram(vertices, 'Mesh', sequence=sequence, std=True)

        # plot sequence diagramms for Angle Error, Joint Error and Mesh Error for every single joint in sequence/joints
        self.plot_sequences(angle, 'Angle', sequence=sequence)
        self.plot_sequences(joint, 'Joint', sequence=sequence)
        self.plot_sequences(vertices, 'Mesh', sequence=sequence)
        
        # plot error over distance
        self.plot_error_distance(angle, distance, 'Angle', sequence=sequence)
        self.plot_error_distance(joint, distance, 'Joint', sequence=sequence)
        
    
    def plot_histogram(self, data, metrics, sequence="", write_csv=False, count=False, std=False):
        data, joint_set = self.reshape_data(data)
        data = data.mean(dim=1)
        data = data.view(-1,)
        if not os.path.isdir(os.path.join(paths['workspace_dir'], 'experiments', self._name, 'evaluation', (sequence if sequence else ""))):
            os.mkdir(os.path.join(paths['workspace_dir'], 'experiments', self._name, 'evaluation', (sequence if sequence else "")))
        path = os.path.join(paths['workspace_dir'], 'experiments', self._name, 'evaluation', (sequence if sequence else ""), metrics + '-hist.png')
        title = metrics +" Error of "+ (sequence if sequence else self._test_data_str) + ", from experiment: " + self.reduced_name()


        if count:
            # Extract the number of values per bin
            fig, ax = plt.subplots()
            n, edges, patches = ax.hist(data.detach().cpu().numpy(), bins=40)
            plt.close(fig)

        # Plot the histogram
        fig, ax = plt.subplots()
        p, edges, patches = ax.hist(data.detach().cpu().numpy(), bins=40, density=True)

        if count:
            for i, rect in enumerate(ax.patches):
                height = rect.get_height()
                ax.annotate(f'{int(n[i])}',xy=(rect.get_x()+rect.get_width()/2, height),
                    xytext=(0,5),textcoords='offset points', ha='center', va='bottom', rotation=90)
            plt.ylim(0,(max(p)+0.02))

        mu = data.mean().detach().cpu().numpy()
        plt.axvline(x=mu, color='b', label=("Mean = "+str(mu.round(2))))
        if std:
            sigma = data.std().detach().cpu().numpy()
            plt.axvline(x=(mu+sigma), color='g', label=("sigma = "+str(sigma.round(2))))
            plt.axvline(x=(mu-sigma), color='g')
        else:
            quantile25 = torch.quantile(data, 0.25).detach().cpu().numpy()
            quantile75 = torch.quantile(data, 0.75).detach().cpu().numpy()
            plt.axvline(x=quantile25, color='g', label=("25%-Quantile = "+str(quantile25.round(2))))
            plt.axvline(x=quantile75, color='g', label=("75%-Quantile = "+str(quantile75.round(2))))
        plt.legend(bbox_to_anchor=(1,1), loc='upper right')
            
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        plt.grid()
        plt.xlabel(("Error in °" if (metrics == "Angle" or metrics == "SIP") else "Error in cm"))
        plt.ylabel("Percentage")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)
        print("Stored histogram as:", path)

        #store data as csv
        if write_csv:
            path = path.replace('.png', '.csv')
            with open((path), 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data.detach().cpu().numpy())

    def plot_boxplot(self, data, metrics, sequence="", write_csv=False, fliers=False):
        data, joint_set = self.reshape_data(data)
        if not os.path.isdir(os.path.join(paths['workspace_dir'], 'experiments', self._name, 'evaluation', (sequence if sequence else ""))):
            os.mkdir(os.path.join(paths['workspace_dir'], 'experiments', self._name, 'evaluation', (sequence if sequence else "")))
        path = os.path.join(paths['workspace_dir'], 'experiments', self._name, 'evaluation', (sequence if sequence else ""), metrics + '-box.png')
        title = metrics +" Error of "+ (sequence if sequence else self._test_data_str) +", from experiment: " + self.reduced_name()

        fig, ax = plt.subplots() 
        plt.boxplot(data.detach().cpu().numpy(), showfliers=fliers)
        ax.set_xticklabels(joint_set, rotation=45)
        plt.xlabel("Jointset ordered by distance fromt root (pelvis)")
        plt.ylabel(("Error in °" if (metrics == "Angle" or metrics == "SIP") else "Error in cm"))
        plt.title(title)
        plt.grid()
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)
        print("Stored boxplot as:", path)

        #store data as csv
        if write_csv:
            path = path.replace('.png', '.csv')
            with open((path), 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                if joint_set:
                    writer.writerow(joint_set)
                writer.writerows(data.detach().cpu().numpy())

    def plot_error_distance(self, error, distance, metrics, sequence="", write_csv=False):
        distance, joint_set = self.reshape_data(distance)
        distance = distance.view(-1,)
        error, joint_set = self.reshape_data(error)
        error = error.view(-1,)

        if not os.path.isdir(os.path.join(paths['workspace_dir'], 'experiments', self._name, 'evaluation', (sequence if sequence else ""))):
            os.mkdir(os.path.join(paths['workspace_dir'], 'experiments', self._name, 'evaluation', (sequence if sequence else ""),))
        path = os.path.join(paths['workspace_dir'], 'experiments', self._name, 'evaluation', (sequence if sequence else ""), metrics + '-err_dist.png')
        title = metrics +" Error over Distance of "+ (sequence if sequence else self._test_data_str) +", from experiment: " + self.reduced_name()

        fig, ax = plt.subplots()
        ax.hist2d(distance.detach().cpu().numpy(), error.detach().cpu().numpy(), bins=(50,50), cmap=cm.jet,)
        plt.grid()
        plt.xlabel("Joint Distance from root")
        plt.ylabel(("Error in °" if (metrics == "Angle" or metrics == "SIP") else "Error in cm"))
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(path)
        plt.close(fig)
        print("Stored plot as:", path)

        #store data as csv
        if write_csv:
            path = path.replace('.png', '.csv')
            with open((path), 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(distance.detach().cpu().numpy())
                writer.writerow(error.detach().cpu().numpy())

    def plot_sequences(self, data, metrics, sequence="", write_csv=False):
        data, joint_set = self.reshape_data(data)
        if not os.path.isdir(os.path.join(paths['workspace_dir'], 'experiments', self._name, 'evaluation', (sequence if sequence else ""), 'sequence')):
            os.mkdir(os.path.join(paths['workspace_dir'], 'experiments', self._name, 'evaluation', (sequence if sequence else ""), 'sequence'))
        
        # Create figure for all joints in jointset
        if (len(data[0]) == 15 or len(data[0]) == 4):
            # plot all figures
            for i in range(len(data[0])):
                path = os.path.join(paths['workspace_dir'], 'experiments', self._name, 'evaluation', (sequence if sequence else ""), 'sequence', joint_set[i] +"_"+metrics+'-seq.png')
                title = joint_set[i] +" "+ metrics +" Error of "+ (sequence if sequence else self._test_data_str) +", from experiment: " + self.reduced_name()
                fig = plt.figure()
                plot_data = data[:,i].detach().cpu().numpy()
                mu = plot_data.mean()
                x_max= plot_data.argmax()
                plt.axhline(y=mu, color='g', label=("Mean = "+str(mu.round(2))))
                plt.scatter(x=x_max, y=plot_data[x_max], color='r', label=("Max = {:.2f} at {:.0f}".format(plot_data.max(), x_max)))
                plt.legend(bbox_to_anchor=(1,1), loc='upper right')
                plt.plot(plot_data, ls= '-')
                plt.grid()
                plt.xlabel("Frames")
                plt.ylabel(("Error in °" if (metrics == "Angle" or metrics == "SIP") else "Error in cm"))
                plt.title(title)
                plt.tight_layout()
                plt.savefig(path)
                plt.close(fig)

                #store data as csv
                if write_csv:
                    path = path.replace('.png', '.csv')
                    with open((path), 'w', encoding='UTF8', newline='') as f:
                        writer = csv.writer(f)
                        if joint_set:
                            writer.writerow(joint_set)
                        writer.writerows(data.detach().cpu().numpy())
                
        else:
            path = os.path.join(paths['workspace_dir'], 'experiments', self._name, 'evaluation', (sequence if sequence else ""), 'sequence', metrics+'-seq.png')
            title = metrics +" Error of "+ (sequence if sequence else self._test_data_str) +", from experiment: " + self.reduced_name()
            fig = plt.figure()
            plot_data = data[:].mean(dim=1).detach().cpu().numpy()
            mu = plot_data.mean()
            x_max= plot_data.argmax()
            plt.axhline(y=mu, color='g', label=("Mean = "+str(mu.round(2))))
            plt.scatter(x=x_max, y=plot_data[x_max], color='r', label=("Max = "+str(plot_data.max().round(2))))
            plt.legend(bbox_to_anchor=(1,1), loc='upper right')
            plt.plot(plot_data, ls= '-')
            plt.grid()
            plt.xlabel("Frames")
            plt.ylabel(("Error in °" if (metrics == "Angle" or metrics == "SIP") else "Error in cm"))
            plt.title(title)
            plt.tight_layout()
            plt.savefig(path)
            plt.close(fig)

        path = os.path.join(paths['workspace_dir'], 'experiments', self._name, 'evaluation', (sequence if sequence else ""), 'sequence')
        print("Stored sequence plot as:", path)
        
        #store data as csv
        if write_csv:
            path = path.replace('.png', '.csv')
            with open((path), 'w', encoding='UTF8', newline='') as f:
                writer = csv.writer(f)
                if joint_set:
                    writer.writerow(joint_set)
                writer.writerows(data.detach().cpu().numpy().mean(dim=1))

    def reduced_name(self):
        #remove Date-Time
        name = self._name[14:]
        # Add further parts that should be cut out of name-string
        for i, part in enumerate(["DIPNet_", "05noise-", "AMASS_", "-Adam", "-AccAuxiliaryLoss"]):
            name = name.replace(part,"")
        return name

    def reshape_data(self, data):
        joint_set = [] 
        if len(data[0][0]) == 4:
            joint_set = ["L-Hip", "R-Hip", "L-Shoul.", "R-Shoul."]
            joint_mask = torch.tensor([1, 2, 16, 17])
            data = torch.stack(data)
            data = data.view(-1,len(joint_set))
        elif len(data[0][0]) == 24:
            joint_set = ["L-Hip", "R-Hip", "Spine1", "L-Knee", "R-Knee", "Spine2", 
            "Spine3", "Neck", "L-Collar", "R-Collar", "Head", "L-Shoul.", "R-Shoul.", "L-Elbow", "R-Elbow"]
            joint_mask = torch.tensor([1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19])
            data = torch.stack(data)
            data = data[:,:, joint_mask]
            data = data.view(-1,len(joint_set))
        elif len(data[0][0]) == 6890:
            data = torch.stack(data)
            data = data.view(-1, 6890)
        else:
            data = torch.stack(data)
            data = data.view(-1,len(data[0][0]))

        return data, joint_set

    def write_to_TensorBoard(self, sip, angle, joint, vertices, path = 'runs/evaluation'):
        # start TensorBoard with "$ tensorboard --logdir=runs"
        writer = SummaryWriter(path)
        self.write_scalars(writer, sip, "sip")
        self.write_scalars(writer, angle, "angle")
        self.write_scalars(writer, joint, "joint")
        self.write_scalars(writer, vertices, "vertices")
        self.write_histogram(writer, sip, "sip")
        self.write_histogram(writer, angle, "angle")
        self.write_histogram(writer, joint, "joints")
        self.write_histogram(writer, vertices, "vertices")
        self.write_scalars_joints(writer, sip, "sip")
        self.write_scalars_joints(writer, angle, "angle")
        self.write_scalars_joints(writer, joint, "joint")
        self.write_scalars_joints(writer, vertices, "vertices")
        self.write_histogram_joints(writer, sip, "sip")
        self.write_histogram_joints(writer, angle, "angle")
        self.write_histogram_joints(writer, joint, "joint")
        self.write_histogram_joints(writer, vertices, "vertices")

    def write_scalars(self, writer, data, metrics):
        for batch in range(len(data)):
            for frame in range(len(data[batch])):
                writer.add_scalar(os.path.join('sequence/', metrics), data[batch][frame].mean(), (batch*300+frame))
    
    def write_scalars_joints(self, writer, data, metrics):
        data, joint_set = self.reshape_data(data)
        for i, name in enumerate(joint_set):
            for frame in range(len(data)):
                writer.add_scalar(os.path.join('detailed sequence ' + metrics, name), data[frame,i], frame)

    def write_histogram(self, writer, data, metrics):
        data = torch.stack(data)
        data = data.view(-1,1)
        writer.add_histogram(os.path.join('histogram/', metrics), data, bins='auto')

    def write_histogram_joints(self, writer, data, metrics):
        data, joint_set = self.reshape_data(data)
        for i, name in enumerate(joint_set):
            writer.add_histogram(os.path.join('detailed histogram ' + metrics, name), data[:,i], bins='auto')
    
    def _prep_sliding_windows(self, x):
        """
        Prepares sliding windows of length (num_past + num_future + 1) for batches of tensors.
        B: batches
        S: sequence length
        F: features

        Args:
            x (torch.Tensor): Tensor to unfold of shape (B * S * F)

        Returns:
            torch.Tensor: Tensor that contains windows of shape (B * S * F * (num_past + num_future + 1))
        """
        return unfold_to_sliding_windows(x, self._num_past_frames, self._num_future_frames)

    @property
    def model_name(self):
        return self._name

    @property
    def simplified_name(self):
        return str(self._model) + "-" + self._train_data_str
