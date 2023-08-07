#!/bin/bash


eval "$(conda shell.bash hook)"
source /home/$USER/miniconda3/etc/profile.d/conda.sh

#conda init bash
conda activate imu2

script="train" # generate-example, evaluate_sequence

python -m hpe_from_imu.${script} -n 20220920_1148-DIPNet_05noise-AMASS_REF-Adam-AccAuxiliaryLoss
python -m hpe_from_imu.${script} -n 20220920_1558-DIPNet_05noise-AMASS_AWI_LKF-Adam-AccAuxiliaryLoss
python -m hpe_from_imu.${script} -n 20220920_2125-DIPNet_05noise-AMASS_AWO_LFT-Adam-AccAuxiliaryLoss
python -m hpe_from_imu.${script} -n 20220920_2129-DIPNet_05noise-AMASS_AMO_LKF-Adam-AccAuxiliaryLoss
python -m hpe_from_imu.${script} -n 20220920_2132-DIPNet_05noise-AMASS_AEO_LKF-Adam-AccAuxiliaryLoss
python -m hpe_from_imu.${script} -n 20220921_1134-DIPNet_05noise-AMASS_AWO_LAO-Adam-AccAuxiliaryLoss 

python -m hpe_from_imu.${script} -n 20220930_1629-DIPNet_05noise-AMASS_spine3-Adam-AccAuxiliaryLoss
python -m hpe_from_imu.${script} -n 20220930_1631-DIPNet_05noise-AMASS_spine3_elbow-Adam-AccAuxiliaryLoss
python -m hpe_from_imu.${script} -n 20220930_1633-DIPNet_05noise-AMASS_collar-Adam-AccAuxiliaryLoss
python -m hpe_from_imu.${script} -n 20220930_1634-DIPNet_05noise-AMASS_shirt-Adam-AccAuxiliaryLoss
python -m hpe_from_imu.${script} -n 20221005_1421-DIPNet_05noise-AMASS_collar_arm-Adam-AccAuxiliaryLoss
python -m hpe_from_imu.${script} -n 20221007_1012-DIPNet_05noise-AMASS_spine3_arm-Adam-AccAuxiliaryLoss

python -m hpe_from_imu.${script} -n 20221006_1207-DIPNet_05noise-AMASS_collar_arm_only-Adam-AccAuxiliaryLoss
python -m hpe_from_imu.${script} -n 20221006_1647-DIPNet_05noise-AMASS_spine3_arm_only-Adam-AccAuxiliaryLoss
python -m hpe_from_imu.${script} -n 20221004_0936-DIPNet_05noise-AMASS_shirt_only-Adam-AccAuxiliaryLoss

python -m hpe_from_imu.${script} -n 20221126_1636-DIPNet_05noise-AMASS_REF-Adam-AccAuxiliaryLoss
python -m hpe_from_imu.${script} -n 20221126_1800-DIPNet_05noise-AMASS_REF-Adam-AccAuxiliaryLoss
python -m hpe_from_imu.${script} -n 20221128_0936-DIPNet_05noise-AMASS_AWO_LFT-Adam-AccAuxiliaryLoss
python -m hpe_from_imu.${script} -n 20221128_1423-DIPNet_05noise-AMASS_collar-Adam-AccAuxiliaryLoss
python -m hpe_from_imu.${script} -n 20221128_1452-DIPNet_05noise-AMASS_collar_arm-Adam-AccAuxiliaryLoss
python -m hpe_from_imu.${script} -n 20221128_1524-DIPNet_05noise-AMASS_collar_arm_only-Adam-AccAuxiliaryLoss