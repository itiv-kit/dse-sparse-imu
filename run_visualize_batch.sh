#!/bin/bash

eval "$(conda shell.bash hook)"
source /home/$USER/miniconda3/etc/profile.d/conda.sh

#conda init bash
conda activate imu2

scenes=("s_10-01_a" "s_10-02" "s_10-03_a" "s_10-04_a" "s_10-05")
#configs=("AWI_LKF" "AMO_LKF" "AEO_LKF" "AWO_LFT" "AWO_LAO") 
configs=("spine3" "spine3_elbow" "spine3_arm" "collar" "collar_arm" "shirt" "collar_arm_only" "spine3_arm_only" "shirt_only")

for scene in "${scenes[@]}"
do
    for config in "${configs[@]}"
    do
        experiment="synth-DIPNet_05noise-AMASS_${config}-offline"
        gt="${scene}_synth-gt"
        example="${scene}_${experiment}"
        python -m hpe_from_imu.visualize_sequence -f $gt ${scene}_synth-DIPNet_05noise-AMASS_REF-offline $example -r 1 1 1 -l -v -c -e -k Xsens DIP_vertices DIP_$config
        mv ./capture/screen_recording.avi ./capture/${scene}_${config}.avi
        echo "renamed video file as: ${scene}_gt-DIP-$config.avi"
    done
done