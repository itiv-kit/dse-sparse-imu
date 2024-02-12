# Design Space Exploration on Efficient and Accurate Human Pose Estimation from Sparse IMU-Sensing

Supporting the design of sparse IMU-sensing systems for human pose estimation, this repository contains a framework to perform a design space exploration on different IMU sensor configurations.
It comprises synthesis of IMU-data, training of a deep neural network and visualization of the human pose estimation.

This work has been published by IEEE and presented on IROS 2023 under the DOI [10.1109/IROS55552.2023.10341256](https://doi.org/10.1109/IROS55552.2023.10341256). Preprint available [here](https://arxiv.org/abs/2308.02397).

# Setup

### Installation Miniconda
Download miniconda from the [official website](https://docs.conda.io/en/latest/miniconda.html).
Open a terminal and go to the downloaded file, set it executable and install.
```
cd DOWNLOAD/PATH
chmod +x MINICONDA-FILE.sh
bash ./MINICONDA-FILE.sh
```

## Python and Conda packages

Only works on Linux.

Create a Conda environment with `Python 3.9`:

```
conda create -n NAME python=3.9
conda activate NAME
```

Install `PyTorch` from pip:

```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

Install `PyTorch3D`:

```
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
```

Install conda packages:

```
bash install.sh
```

Install pip packages:
```
pip install -r requirements.txt
```
Install `Open3D` (used for visualization), `tqdm` (progress meter), `smplx` (SMPL body model loader), `Poutyne` (training framework), `chumpy`, `nvidia-ml-py` (NVIDIA management library), `einops` (einops tensor operations), `timm` (PyTorch Image Models - used for transformers), `torchinfo` (model summary):

## Setup Datasets

Setup a working directory in the location of your choice (change `working_dir` in `hpe_from_imu/configuration.py`). This directory should contain the directories `body_model`, `configurations`, `example`, `experiments` and `hpe_from_imu`:

```
Human_Pose_Estimation_System/
├── body_model/
├── configurations/
├── example/
├── experiments/
└── hpe_from_imu
```

Setup a data directory in the location of your choice (but change `data_dir` in `hpe_from_imu/configuration.py`). This directory should contain the directories `dataset_raw` and `dataset_work`:

```
data/
├── dataset_raw/
└── dataset_work/
```

Download the SMPL model files from the [SMPL project page](https://smpl.is.tue.mpg.de/download.php). Choose `version 1.0.0 for Python 2.7 (10 shape PCs)`. Unzip, move the two models `*.pkl` directly into the `data/body_model/` folder and rename them to `SMPL_male.pkl` and `SMPL_female.pkl` respectively (or change the config):

```
└── body_model/
    ├── SMPL_male.pkl
    └── SMPL_female.pkl
```

Download the DIP-IMU and the synthetic AMASS dataset from the [DIP project page](https://dip.is.tuebingen.mpg.de/download.php). Download `DIP IMU AND OTHERS` and `SYNTHETIC AMASS 60FPS`. Unzip them (and contained zips) and move them to `dataset_raw`.

Download AMASS datasets from the [AMASS project page](https://amass.is.tue.mpg.de/download.php). You should choose the `SMPL+H G` option because only standard SMPL parameters are used. Unzip the datasets into the AMASS folder.

Download the TotalCapture dataset from their [project page](https://cvssp.org/data/totalcapture/) and unzip into the TotalCapture folder.

You can ask the authors of the `Deep Inertial Pose` paper for a preprocessed TotalCapture dataset.

Your `dataset_raw` directory should now look like this:

```
└── dataset_raw/
    ├── AMASS
    │   ├── ACCAD
    │   │   ├── Female1General_c3d 
    │   │   └── ....
    │   ├── BMLmovi
    │   ├── ...
    │   └── Transitions_mocap
    ├── AMASS_TotalCapture
    │   ├── s1
    │   ├── ...
    │   └── s5
    ├── DIP_IMU_and_Others
    │   ├── DIP_IMU
    │   │   ├── s_01
    │   │   ├── ...
    │   │   └── s_10
    │   └── DIP_IMU_nn
    ├── Synthetic_60FPS
    │   ├── AMASS_ACCAD
    │   ├── ...
    │   └── JointLimit
    └── TotalCapture
        ├── s1
        ├── ...
        └── s5
```
## Configuration

As a first step customize the path configuration in [hpe_from_imu/configuration.py](hpe_from_imu/configuration.py) so that all paths are set correctly according to your setup. Then run
```
python -m hpe_from_imu.configuration
```
to write to [configuration/config.yaml](configuration/config.yaml) which is used throughout the whole project.

## Preprocessing and synthesis

Customize the main of `preprocess.py` to define which preprocessing and synthesis scripts you want to run. And then start the preprocessing script.
```
python -m hpe_from_imu.preprocess
```
This can take a while but you should be able to see the progress in the terminal.

A sensible test run could include these 4 steps to check if preprocess, synthesis and example preparation all work:
```python
if __name__ == '__main__':
    preprocess_DIP_IMU()
    synthesize_DIP()
    preprocess_DIP_TP_synth()
    prep_DIP_example("s_10", "02")
```
If you want to synthesize custom configurations, execute those steps:
```python
if __nam__ == '__main__':
    custom_synthesize_AMASS("AMASS_dse_complete", "dse_complete", "dse_complete")
    preprocess_AMASS_TP_custom_synth("AMASS_dse_complete", "AMASS_dse_complete_preprocessed")
    custom_synthesize_DIP("DIP_dse_complete", "dse_complete", "dse_complete")
    preprocess_DIP_TP_custom_synth("DIP_dse_complete", "DIP_dse_complete_preprocessed")
```

## Training a model, evaluating a model

During training checkpoints and statistics get tracked that need a folder to be stored in. Create folder [experiments/](experiments/) as collection for any other training experiments.

```
mkdir ./experiments/
```

Before you start training a model, have a quick look into [configuration/experiments/](configuration/experiments/). This directory contains several experiment configs. An experiment is basically the combination of a model definition, training parameters, used datasets and required pre- and postprocessing steps (which are different to the preprocessing performed before).

All experiment configs are based on [base.yml](configuration/experiments/base.yml) and are then combined with a second `CONFIGURATION.yml` which you choose. Configurations defined in this second config file overwrite the base configuration.

The script [train.py](hpe_from_imu/train.py) is not only used for training models but also for evaluating their latency, power draw or accuracy metrics. To change which of these training steps are performed check the main function:
```python
if __name__ == '__main__':
    ...
    experiment.evaluate_latency((26, 72))
    experiment.evaluate_power_draw((26, 72))
    experiment.train(1)
    experiment.evaluate_online("DIP_test")
```
This would perform latency and power draw evaluations, train the model for 1 epoch and then evaluate the online performance on the test partition of the DIP-IMU dataset.

An experiment can be started from scratch with a config file or resumed if you know its name. Let's use [dipnet_DIP_train.yml](configuration/experiments/dipnet_DIP_train.yml) for this example.
```
python -m hpe_from_imu.train -c dipnet_DIP_train.yml
```

Output (shortened):
```
Epoch: 1/1 Train steps: 53 9.37s loss: 0.046000                                                   
lr: 0.001, loss: 0.046, val_loss: nan
Prediction steps: 11 1.11s                                               
100%|█████████████████| 166/166 [00:27<00:00,  5.97it/s]
SIP Error (deg): 17.26 (+/- 12.91)
Angular Error (deg): 19.98 (+/- 13.36)
Positional Error (cm): 8.24 (+/- 6.47)
Mesh Error (cm): 10.60 (+/- 7.72)
Jitter Error (100m/s^3): 1373.32 (+/- 663.08)
```

The `-c` flag is used to define which config file to use. The script will look in [configuration/experiments/](configuration/experiments/) for the named config. The experiment now runs all the steps and outputs the progress and results into the terminal. Most of the information will also be logged and saved in a newly created folder in [experiments/](experiments/). The correct folder name should begin with the starting time of the experiment and include the name of the model architecture (`DIPNet`), the dataset used for training (`DIP_train`), the used optimizer (`ADAM`) aswell as the used loss function (`mseloss`). This folder contains the experiment config, tensorboard logs, automatically generated plots of the training process, model weights, etc.

If we want to resume this experiment at some point we can use the `-n` flag with the name of the folder. 
```
python -m hpe_from_imu.train -n DATE_TIME-DIPNet-DIP_train-Adam-mse
```
The model then loads the weights and training or evaluation can be resumed.

## Generating an example

If you possibly want to visualize the predictions made by a trained model, use the [generate-example.py](generate-example.py) script to save the predictions in a format that can be read by the [visualize script](visualize_sequence.py). Check the script for details how to generate predictions for a different example.

Lets use the model we trained in the last step for the prediction:
```
python -m hpe_from_imu.generate-example -n DATE_TIME-DIPNet_DIP_train-DIP_train-Adam-ms
```
Output (shortened):
```
Saved predictions under s_10-02-DIPNet_DIP_train-DIP_train
```
The script saved the predictions aswell as ground-truth in `data/examples`.

## Visualization

We can now visualize the predictions with [visualize_sequence.py](visualize_sequence.py). Run:
```
python -m hpe_from_imu.visualize_sequence -f s_10-02-gt s_10-02-DIPNet-DIP_train-online -r 1 1 -v
```
The `-f` flag is used to define which files should be visualized. The scripts looks in the `data/examples` directory and only considers files that end on `-pose.pt`. In our case `s_10-02-gt` opens `s_10-02-gt-pose.pt`. The `-r` flag requires a 1 or 0 for each given file to show whether the SMPL parameters are stored as a rotation matrix or in axis-angle format. 1 for rotation matrix, 0 for axis-angle. The `-v` flag is used to increase the verbosity of the output. For additional flags please refer to [visualize_sequence.py](visualize_sequence.py#L68).


# Used datasets

Data sets and body models have to be downloaded from source:

- [SMPL](https://smpl.is.tue.mpg.de/modellicense.html)
- [DIP-IMU](https://dip.is.tuebingen.mpg.de/license.html)
- [AMASS](https://amass.is.tue.mpg.de/license.html)
- [Total Capture](https://cvssp.org/data/totalcapture/)

# Citing this work

If you found this tool useful, please use the following bibtex to cite us

```
@inproceedings{Fuerst-Walter2023,
    Title = {Design Space Exploration on Efficient and Accurate Human Pose Estimation from Sparse IMU-Sensing},
    Author = {Fuerst-Walter, Iris and Nappi, Antonio and Harbaum, Tanja and Becker, Juergen},
    Booktitle = IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS),
    Year = {in press 2023},
}
```

# License
This repository is under MIT License. Please see the [full license here](LICENSE).
