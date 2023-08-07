from os.path import join as join_path
import yaml

# run script once to write 'config.yaml' file with updated paths 
# edit 'data_dir' to the lacation of your data storage

class Constants():
    config_path = "configuration/config.yml"


class _Path():
    """
    Class that contains all paths for directories and files required for this project.
    """
    # Paths for general directories
    workspace_dir = "/home/$USER/HPE"
    shared_data_dir = "/tools/datasets/IMU"
    data_dir =  "/home/$USER/HPE"
    data_dir_raw = join_path(data_dir, "dataset_raw")
    data_dir_work = join_path(data_dir, "dataset_work")

    # Paths for DIP-IMU dataset
    DIP_IMU = join_path(shared_data_dir, "DIP_IMU_and_Others/DIP_IMU")
    DIP_IMU_preprocessed = join_path(data_dir_work, "DIP_IMU")
    DIP_IMU_synth = join_path(data_dir_raw, "DIP_IMU_synth")
    DIP_IMU_synth_joints = join_path(DIP_IMU_synth, "joint.pt")
    DIP_IMU_synth_poses = join_path(DIP_IMU_synth, "pose.pt")
    DIP_IMU_synth_shapes = join_path(DIP_IMU_synth, "shape.pt")
    DIP_IMU_synth_trans = join_path(DIP_IMU_synth, "tran.pt")
    DIP_IMU_synth_vaccs = join_path(DIP_IMU_synth, "vacc.pt")
    DIP_IMU_synth_vrots = join_path(DIP_IMU_synth, "vrot.pt")

    DIP_IMU_REF_preprocessed = join_path(data_dir_work, "DIP_IMU_Refernce")
    DIP_IMU_AWO_LFT_preprocessed = join_path(data_dir_work, "DIP_IMU_AWO_LFT")
    DIP_IMU_collar_preprocessed =  join_path(data_dir_work, "DIP_IMU_collar")
    DIP_IMU_collar_arm_preprocessed = join_path(data_dir_work, "DIP_IMU_collar_arm")
    DIP_IMU_collar_arm_only_preprocessed = join_path(data_dir_work, "DIP_IMU_collar_arm_only")
    

    DIP_IMU_synth_preprocessed = join_path(data_dir_work, "DIP_IMU_synth")

    # Paths for AMASS dataset
    AMASS_raw = join_path(shared_data_dir, "AMASS")
    AMASS_DIP = join_path(shared_data_dir, "Synthetic_60FPS")
    AMASS_TP = join_path(shared_data_dir, "AMASS_TP_synth")
    AMASS_TP_joints = join_path(AMASS_TP, "joint.pt")
    AMASS_TP_poses = join_path(AMASS_TP, "pose.pt")
    AMASS_TP_shapes = join_path(AMASS_TP, "shape.pt")
    AMASS_TP_trans = join_path(AMASS_TP, "tran.pt")
    AMASS_TP_vaccs = join_path(AMASS_TP, "vacc.pt")
    AMASS_TP_vrots = join_path(AMASS_TP, "vrot.pt")

    AMASS_DIP_preprocessed = join_path(data_dir_work, "AMASS_DIP")
    AMASS_TP_preprocessed = join_path(data_dir_work, "AMASS_TP")

    # Paths for TotalCapture dataset
    TC_raw = join_path(shared_data_dir, "TotalCapture")
    TC_DIP = join_path(shared_data_dir, "TotalCapture_60FPS_Original")
    TC_DIP_preprocessed = join_path(data_dir_work, "TotalCapture")

    # Paths for generated examples
    #example_dir = join_path(data_dir, "example/")
    example_dir = join_path(workspace_dir, "example/")

    # Paths for SMPL files
    #SMPL_dir = join_path(data_dir, "body_model/")
    SMPL_dir = join_path(workspace_dir, "body_model/")
    SMPL_female = join_path(SMPL_dir, "SMPL_female.pkl")
    SMPL_male = join_path(SMPL_dir, "SMPL_male.pkl")
    SMPLH_male = join_path(SMPL_dir, "SMPLH_MALE.pkl")
    SMPLH_female = join_path(SMPL_dir, "SMPLH_FEMALE.pkl")
    SMPLH_neutral = join_path(SMPL_dir, "SMPLH_NEUTRAL.pkl")

    # Paths for experiments
    experiments = join_path(workspace_dir, "experiments/")
    experiments_config = "./configuration/experiments"
    experiments_base_config = join_path(experiments_config, "base.yml")

    # Paths for TransPose model training
    TP_training = join_path(data_dir_work, "TP_training")
    TP_training_S1 = join_path(TP_training, "S1")
    TP_training_S2 = join_path(TP_training, "S2")
    TP_training_S3 = join_path(TP_training, "S3")
    TP_training_total = join_path(TP_training, "total")

    # Paths to store custom vertex and joint datasets
    Custom_DIP_path = join_path(data_dir_raw, "DIP_IMU_Custom_1")
    Custom_DIP_path_preprocessed = join_path(data_dir_work, "DIP_IMU_Custom_1")
    Custom_AMASS_path = join_path(data_dir_raw, "AMASS_Custom_1")
    Custom_AMASS_path_preprocessed = join_path(data_dir_work, "AMASS_Custom_1")

    # NodePositions
    DIP_REF = join_path(data_dir_raw, "DIP_Refernce")
    DIP_REF_preprocessed = join_path(data_dir_work, "DIP_Refernce")
    AMASS_REF = join_path(data_dir_raw, "AMASS_Refernce")
    AMASS_REF_preprocessed = join_path(data_dir_work, "AMASS_Refernce")

    DIP_AWI_LKF = join_path(data_dir_raw, "DIP_AWI-LKF")
    DIP_AWI_LKF_preprocessed = join_path(data_dir_work, "DIP_AWI-LKF")
    AMASS_AWI_LKF = join_path(data_dir_raw, "AMASS_AWI-LKF")
    AMASS_AWI_LKF_preprocessed = join_path(data_dir_work, "AMASS_AWI-LKF")

    DIP_AMO_LKF = join_path(data_dir_raw, "DIP_AMO-LKF")
    DIP_AMO_LKF_preprocessed = join_path(data_dir_work, "DIP_AMO-LKF")
    AMASS_AMO_LKF = join_path(data_dir_raw, "AMASS_AMO-LKF")
    AMASS_AMO_LKF_preprocessed = join_path(data_dir_work, "AMASS_AMO-LKF")

    DIP_AEO_LKF = join_path(data_dir_raw, "DIP_AEO-LKF")
    DIP_AEO_LKF_preprocessed = join_path(data_dir_work, "DIP_AEO-LKF")
    AMASS_AEO_LKF = join_path(data_dir_raw, "AMASS_AEO-LKF")
    AMASS_AEO_LKF_preprocessed = join_path(data_dir_work, "AMASS_AEO-LKF")

    DIP_AWO_LAO = join_path(data_dir_raw, "DIP_AWO-LAO")
    DIP_AWO_LAO_preprocessed = join_path(data_dir_work, "DIP_AWO-LAO")
    AMASS_AWO_LAO = join_path(data_dir_raw, "AMASS_AWO-LAO")
    AMASS_AWO_LAO_preprocessed = join_path(data_dir_work, "AMASS_AWO-LAO")

    DIP_AWO_LFT = join_path(data_dir_raw, "DIP_AWO-LFT")
    DIP_AWO_LFT_preprocessed = join_path(data_dir_work, "DIP_AWO-LFT")
    AMASS_AWO_LFT = join_path(data_dir_raw, "AMASS_AWO-LFT")
    AMASS_AWO_LFT_preprocessed = join_path(data_dir_work, "AMASS_AWO-LFT")

    # Sensor Configuration
    DIP_spine3 = join_path(data_dir_raw, "DIP_spine3")
    DIP_spine3_preprocessed = join_path(data_dir_work, "DIP_spine3")
    AMASS_spine3 = join_path(data_dir_raw, "AMASS_spine3")
    AMASS_spine3_preprocessed = join_path(data_dir_work, "AMASS_spine3")

    DIP_spine3_arm = join_path(data_dir_raw, "DIP_spine3_arm")
    DIP_spine3_arm_preprocessed = join_path(data_dir_work, "DIP_spine3_arm")
    AMASS_spine3_arm = join_path(data_dir_raw, "AMASS_spine3_arm")
    AMASS_spine3_arm_preprocessed = join_path(data_dir_work, "AMASS_spine3_arm")

    DIP_spine3_arm_only = join_path(data_dir_raw, "DIP_spine3_arm_only")
    DIP_spine3_arm_only_preprocessed = join_path(data_dir_work, "DIP_spine3_arm_only")
    AMASS_spine3_arm_only = join_path(data_dir_raw, "AMASS_spine3_arm_only")
    AMASS_spine3_arm_only_preprocessed = join_path(data_dir_work, "AMASS_spine3_arm_only")

    DIP_spine3_elbow = join_path(data_dir_raw, "DIP_spine3_elbow")
    DIP_spine3_elbow_preprocessed = join_path(data_dir_work, "DIP_spine3_elbow")
    AMASS_spine3_elbow = join_path(data_dir_raw, "AMASS_spine3_elbow")
    AMASS_spine3_elbow_preprocessed = join_path(data_dir_work, "AMASS_spine3_elbow")

    DIP_collar = join_path(data_dir_raw, "DIP_collar")
    DIP_collar_preprocessed = join_path(data_dir_work, "DIP_collar")
    AMASS_collar = join_path(data_dir_raw, "AMASS_collar")
    AMASS_collar_preprocessed = join_path(data_dir_work, "AMASS_collar")

    DIP_collar_arm = join_path(data_dir_raw, "DIP_collar_arm")
    DIP_collar_arm_preprocessed = join_path(data_dir_work, "DIP_collar_arm")
    AMASS_collar_arm = join_path(data_dir_raw, "AMASS_collar_arm")
    AMASS_collar_arm_preprocessed = join_path(data_dir_work, "AMASS_collar_arm")

    DIP_collar_arm_only = join_path(data_dir_raw, "DIP_collar_arm_only")
    DIP_collar_arm_only_preprocessed = join_path(data_dir_work, "DIP_collar_arm_only")
    AMASS_collar_arm_only = join_path(data_dir_raw, "AMASS_collar_arm_only")
    AMASS_collar_arm_only_preprocessed = join_path(data_dir_work, "AMASS_collar_arm_only")

    DIP_shirt = join_path(data_dir_raw, "DIP_shirt")
    DIP_shirt_preprocessed = join_path(data_dir_work, "DIP_shirt")
    AMASS_shirt = join_path(data_dir_raw, "AMASS_shirt")
    AMASS_shirt_preprocessed = join_path(data_dir_work, "AMASS_shirt")

    DIP_shirt_only = join_path(data_dir_raw, "DIP_shirt_only")
    DIP_shirt_only_preprocessed = join_path(data_dir_work, "DIP_shirt_only")
    AMASS_shirt_only = join_path(data_dir_raw, "AMASS_shirt_only")
    AMASS_shirt_only_preprocessed = join_path(data_dir_work, "AMASS_shirt_only")

    DIP_dse_complete = join_path(data_dir_raw, "DIP_dse_complete")
    DIP_dse_complete_preprocessed = join_path(data_dir_work, "DIP_dse_complete")
    AMASS_dse_complete = join_path(data_dir_raw, "AMASS_dse_complete")
    AMASS_dse_complete_preprocessed = join_path(data_dir_work, "AMASS_dse_complete")

class _Preprocessing():
    """
    Class that contains relevant defintions for the preprocessing script.
    """
    # corrected order of Sensors: (https://github.com/eth-ait/dip18/issues/16)
    # [0: head, 1: pelvis, 2: sternum, 3: lhand, 4: rhand, 5: lshoulder, 6: rshoulder, 7: larm, 8: rarm, 
    #   9: lhip, 10: rhip, 11: lknee, 12: rknee, 13: lwrist, 14: rwrist, 15: lfoot, 16: rfoot]
    DIP_IMU_mask = [7, 8, 11, 12, 0, 1]

    # Sensor Position: Joint Positions
    DIP_IMU_AWO_LFT =[7, 8, 15, 16, 0, 1]  
    # Sensor Configuration Joint Positions
    DIP_IMU_collar = [13, 14, 11, 12, 5, 6, 1]
    DIP_IMU_collar_arm = [13, 14, 7, 8, 11, 12, 5, 6, 1]
    DIP_IMU_collar_arm_only = [13, 14, 7, 8, 5, 6, 1]

    DIP_test_split = ['s_09', 's_10']
    # Validation split is self defined - no information about DIP validation split available
    DIP_validation_split = ['s_0102.pkl','s_0305.pkl','s_0501_b.pkl', 's_0703.pkl', 's_0804.pkl']

    AMASS_dataset = [ # usded subset of AMASS
        'ACCAD',
        'BioMotionLab_NTroje',
        'BMLhandball',
        'BMLmovi',
        'CMU',
        'DFaust_67',
        'DanceDB',
        'EKUT',
        'Eyes_Japan_Dataset',
        # 'GRAB',
        'HUMAN4D',
        'HumanEva',
        'KIT',
        'MPI_HDM05',
        'MPI_Limits',
        'MPI_mosh',
        'SFU',
        # 'SOMA',
        'SSM_synced',
        'TCD_handMocap',
        # 'TotalCapture',
        'Transitions_mocap']

    AMASS_test_split = []

    #[0: Head, 1: Sternum (Spine3), 2: Pelvis (Hips), 3: L_UpArm, 4: R_UpArm, 5: L_LowArm, 6: R_LowArm, 
    # 7: L_UpLeg, 8: R_UpLeg, 9: L_LowLeg, 10: R_LowLeg, 11: L_Foot, 12: R_Foot]
    # Source: https://github.com/zhezh/TotalCapture-Toolbox/blob/master/gendata/config.yaml 
    TotalCapture_mask = [2, 3, 0, 1, 4, 5] 


class _TP_joint_set():
    """
    Class that contains joint definitions for SMPL models.
    From https://github.com/Xinyu-Yi/TransPose 
    """
    # leaf = [7, 8, 12, 20, 21] # Incorrect
    leaf = [7, 8, 15, 20, 21] # TransPose/PIP refer "Head" as leaf-joint, not "Neck"
    full = list(range(1, 24))
    reduced = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
    ignored = [0, 7, 8, 10, 11, 20, 21, 22, 23]

    lower_body = [0, 1, 2, 4, 5, 7, 8, 10, 11]
    lower_body_parent = [None, 0, 0, 1, 2, 3, 4, 5, 6]

    n_leaf = len(leaf)
    n_full = len(full)
    n_reduced = len(reduced)
    n_ignored = len(ignored)


class _DATASET_PATHS():
    DIP_train = join_path(_Path.DIP_IMU_preprocessed, "300-train.pt")
    DIP_test = join_path(_Path.DIP_IMU_preprocessed, "300-test.pt")
    AMASS_DIP = join_path(_Path.AMASS_DIP_preprocessed, "300-train.pt")
    AMASS_TP = join_path(_Path.AMASS_TP_preprocessed, "300-train.pt")
    TC = join_path(_Path.TC_DIP_preprocessed, "300-test.pt")
    DIP_synth = join_path(_Path.DIP_IMU_synth_preprocessed, "300-train.pt")
    TP_training = join_path(_Path.TP_training, "300-train.pt")
    TP_training_S1 = join_path(_Path.TP_training_S1, "300-train.pt")
    TP_training_S2 = join_path(_Path.TP_training_S2, "300-train.pt")
    TP_training_S3 = join_path(_Path.TP_training_S3, "300-train.pt")
    TP_training_total = join_path(_Path.TP_training_total, "300-train.pt")
      
    # Node positions
    AMASS_REF = join_path(_Path.AMASS_REF_preprocessed, "300-train.pt")
    DIP_REF_train = join_path(_Path.DIP_REF_preprocessed, "300-train.pt")
    DIP_REF_test = join_path(_Path.DIP_REF_preprocessed, "300-test.pt")

    DIP_IMU_REF_train = join_path(_Path.DIP_IMU_REF_preprocessed, "300-train.pt")
    DIP_IMU_REF_test = join_path(_Path.DIP_IMU_REF_preprocessed, "300-test.pt")
    DIP_IMU_REF_validation = join_path(_Path.DIP_IMU_REF_preprocessed, "300-validation.pt")

    AMASS_AWI_LKF = join_path(_Path.AMASS_AWI_LKF_preprocessed, "300-train.pt")
    DIP_AWI_LKF_train = join_path(_Path.DIP_AWI_LKF_preprocessed, "300-train.pt")
    DIP_AWI_LKF_test = join_path(_Path.DIP_AWI_LKF_preprocessed, "300-test.pt")

    AMASS_AMO_LKF = join_path(_Path.AMASS_AMO_LKF_preprocessed, "300-train.pt")
    DIP_AMO_LKF_train = join_path(_Path.DIP_AMO_LKF_preprocessed, "300-train.pt")
    DIP_AMO_LKF_test = join_path(_Path.DIP_AMO_LKF_preprocessed, "300-test.pt")
    
    AMASS_AEO_LKF = join_path(_Path.AMASS_AEO_LKF_preprocessed, "300-train.pt")
    DIP_AEO_LKF_train = join_path(_Path.DIP_AEO_LKF_preprocessed, "300-train.pt")
    DIP_AEO_LKF_test = join_path(_Path.DIP_AEO_LKF_preprocessed, "300-test.pt")

    AMASS_AWO_LAO = join_path(_Path.AMASS_AWO_LAO_preprocessed, "300-train.pt")
    DIP_AWO_LAO_train = join_path(_Path.DIP_AWO_LAO_preprocessed, "300-train.pt")
    DIP_AWO_LAO_test = join_path(_Path.DIP_AWO_LAO_preprocessed, "300-test.pt")

    AMASS_AWO_LFT = join_path(_Path.AMASS_AWO_LFT_preprocessed, "300-train.pt")
    DIP_AWO_LFT_train = join_path(_Path.DIP_AWO_LFT_preprocessed, "300-train.pt")
    DIP_AWO_LFT_test = join_path(_Path.DIP_AWO_LFT_preprocessed, "300-test.pt")

    DIP_IMU_AWO_LFT_train = join_path(_Path.DIP_IMU_AWO_LFT_preprocessed, "300-train.pt")
    DIP_IMU_AWO_LFT_test = join_path(_Path.DIP_IMU_AWO_LFT_preprocessed, "300-test.pt")
    DIP_IMU_AWO_LFT_validation = join_path(_Path.DIP_IMU_AWO_LFT_preprocessed, "300-validation.pt")

    # Sensor Configuration
    AMASS_spine3 = join_path(_Path.AMASS_spine3_preprocessed, "300-train.pt")
    DIP_spine3_train = join_path(_Path.DIP_spine3_preprocessed, "300-train.pt")
    DIP_spine3_test = join_path(_Path.DIP_spine3_preprocessed, "300-test.pt")

    AMASS_spine3_arm = join_path(_Path.AMASS_spine3_arm_preprocessed, "300-train.pt")
    DIP_spine3_arm_train = join_path(_Path.DIP_spine3_arm_preprocessed, "300-train.pt")
    DIP_spine3_arm_test = join_path(_Path.DIP_spine3_arm_preprocessed, "300-test.pt")

    AMASS_spine3_arm_only = join_path(_Path.AMASS_spine3_arm_only_preprocessed, "300-train.pt")
    DIP_spine3_arm_only_train = join_path(_Path.DIP_spine3_arm_only_preprocessed, "300-train.pt")
    DIP_spine3_arm_only_test = join_path(_Path.DIP_spine3_arm_only_preprocessed, "300-test.pt")

    AMASS_spine3_elbow = join_path(_Path.AMASS_spine3_elbow_preprocessed, "300-train.pt")
    DIP_spine3_elbow_train = join_path(_Path.DIP_spine3_elbow_preprocessed, "300-train.pt")
    DIP_spine3_elbow_test = join_path(_Path.DIP_spine3_elbow_preprocessed, "300-test.pt")

    AMASS_collar = join_path(_Path.AMASS_collar_preprocessed, "300-train.pt")
    DIP_collar_train = join_path(_Path.DIP_collar_preprocessed, "300-train.pt")
    DIP_collar_test = join_path(_Path.DIP_collar_preprocessed, "300-test.pt")

    DIP_IMU_collar_train = join_path(_Path.DIP_IMU_collar_preprocessed, "300-train.pt")
    DIP_IMU_collar_test = join_path(_Path.DIP_IMU_collar_preprocessed, "300-test.pt")
    DIP_IMU_collar_validation = join_path(_Path.DIP_IMU_collar_preprocessed, "300-validation.pt")

    AMASS_collar_arm = join_path(_Path.AMASS_collar_arm_preprocessed, "300-train.pt")
    DIP_collar_arm_train = join_path(_Path.DIP_collar_arm_preprocessed, "300-train.pt")
    DIP_collar_arm_test = join_path(_Path.DIP_collar_arm_preprocessed, "300-test.pt")

    DIP_IMU_collar_arm_train = join_path(_Path.DIP_IMU_collar_arm_preprocessed, "300-train.pt")
    DIP_IMU_collar_arm_test = join_path(_Path.DIP_IMU_collar_arm_preprocessed, "300-test.pt")
    DIP_IMU_collar_arm_validation = join_path(_Path.DIP_IMU_collar_arm_preprocessed, "300-validation.pt")

    AMASS_collar_arm_only = join_path(_Path.AMASS_collar_arm_only_preprocessed, "300-train.pt")
    DIP_collar_arm_only_train = join_path(_Path.DIP_collar_arm_only_preprocessed, "300-train.pt")
    DIP_collar_arm_only_test = join_path(_Path.DIP_collar_arm_only_preprocessed, "300-test.pt")

    DIP_IMU_collar_arm_only_train = join_path(_Path.DIP_IMU_collar_arm_only_preprocessed, "300-train.pt")
    DIP_IMU_collar_arm_only_test = join_path(_Path.DIP_IMU_collar_arm_only_preprocessed, "300-test.pt")
    DIP_IMU_collar_arm_only_validation = join_path(_Path.DIP_IMU_collar_arm_only_preprocessed, "300-validation.pt")

    AMASS_shirt = join_path(_Path.AMASS_shirt_preprocessed, "300-train.pt")
    DIP_shirt_train = join_path(_Path.DIP_shirt_preprocessed, "300-train.pt")
    DIP_shirt_test = join_path(_Path.DIP_shirt_preprocessed, "300-test.pt")

    AMASS_shirt_only = join_path(_Path.AMASS_shirt_only_preprocessed, "300-train.pt")
    DIP_shirt_only_train = join_path(_Path.DIP_shirt_only_preprocessed, "300-train.pt")
    DIP_shirt_only_test = join_path(_Path.DIP_shirt_only_preprocessed, "300-test.pt")

    AMASS_dse_complete = join_path(_Path.AMASS_dse_complete_preprocessed, "300-train.pt")
    DIP_dse_complete_train = join_path(_Path.DIP_dse_complete_preprocessed, "300-train.pt")
    DIP_dse_complete_test = join_path(_Path.DIP_dse_complete_preprocessed, "300-test.pt")

dataset_base = {
    "DIP_test": "DIP",
    "DIP_train": "DIP",
    "DIP_synth": "DIP_synth",
    "AMASS_DIP": "AMASS_DIP",
    "AMASS_TP": "AMASS_TP",
    "TC": "TC",
    "TP_training_S1": "TP_training_S1",
    "TP_training_S2": "TP_training_S2",
    "TP_training_S3": "TP_training_S3",
    "TP_training_total": "TP_training_total",

    # Node Positions
    "AMASS_REF" : "AMASS_TP",
    "DIP_REF_train" : "DIP",
    "DIP_REF_test" : "DIP",

    "DIP_IMU_REF_train" : "DIP",
    "DIP_IMU_REF_test" : "DIP",
    "DIP_IMU_REF_validation" : "DIP",

    "AMASS_AWI_LKF" : "AMASS_TP",
    "DIP_AWI_LKF_train" : "DIP",
    "DIP_AWI_LKF_test" : "DIP",

    "AMASS_AMO_LKF" : "AMASS_TP",
    "DIP_AMO_LKF_train" : "DIP",
    "DIP_AMO_LKF_test" : "DIP",
    
    "AMASS_AEO_LKF" : "AMASS_TP",
    "DIP_AEO_LKF_train" : "DIP",
    "DIP_AEO_LKF_test" : "DIP",

    "AMASS_AWO_LAO" : "AMASS_TP",
    "DIP_AWO_LAO_train" : "DIP",
    "DIP_AWO_LAO_test" : "DIP",

    "AMASS_AWO_LFT" : "AMASS_TP",
    "DIP_AWO_LFT_train" : "DIP",
    "DIP_AWO_LFT_test" : "DIP",

    "DIP_IMU_AWO_LFT_train" : "DIP",
    "DIP_IMU_AWO_LFT_test" : "DIP",
    "DIP_IMU_AWO_LFT_validation" : "DIP",

    # Sensor Configuration
    "AMASS_spine3" : "AMASS_TP",
    "DIP_spine3_train" : "DIP",
    "DIP_spine3_test" : "DIP",

    "AMASS_spine3_arm" : "AMASS_TP",
    "DIP_spine3_arm_train" : "DIP",
    "DIP_spine3_arm_test" : "DIP",

    "AMASS_spine3_arm_only" : "AMASS_TP",
    "DIP_spine3_arm_only_train" : "DIP",
    "DIP_spine3_arm_only_test" : "DIP",

    "AMASS_spine3_elbow" : "AMASS_TP",
    "DIP_spine3_elbow_train" : "DIP",
    "DIP_spine3_elbow_test" : "DIP",

    "AMASS_collar" : "AMASS_TP",
    "DIP_collar_train" : "DIP",
    "DIP_collar_test" : "DIP",

    "DIP_IMU_collar_train" : "DIP",
    "DIP_IMU_collar_test" : "DIP",
    "DIP_IMU_collar_validation" : "DIP",

    "AMASS_collar_arm" : "AMASS_TP",
    "DIP_collar_arm_train" : "DIP",
    "DIP_collar_arm_test" : "DIP",

    "DIP_IMU_collar_arm_train" : "DIP",
    "DIP_IMU_collar_arm_test" : "DIP",
    "DIP_IMU_collar_arm_validation" : "DIP", 

    "AMASS_collar_arm_only" : "AMASS_TP",
    "DIP_collar_arm_only_train" : "DIP",
    "DIP_collar_arm_only_test" : "DIP",

    "DIP_IMU_collar_arm_only_train" : "DIP",
    "DIP_IMU_collar_arm_only_test" : "DIP",
    "DIP_IMU_collar_arm_only_validation" : "DIP",   

    "AMASS_shirt" : "AMASS_TP",
    "DIP_shirt_train" : "DIP",
    "DIP_shirt_test" : "DIP",

    "AMASS_shirt_only" : "AMASS_TP",
    "DIP_shirt_only_train" : "DIP",
    "DIP_shirt_only_test" : "DIP",
    
    "AMASS_dse_complete" : "AMASS_TP",
    "DIP_dse_complete_train" : "DIP",
    "DIP_dse_complete_test" : "DIP",
}

class _JOINT_CONFIG():
    """ 
    Class that contains joint configurations for synthesizing.
    Each vertex reflects a point on the SMPL-mesh (NOT SMPLX-mesh)
    that is equippet with a virtual IMU-sensor.
    """
    # Xsens = [sternum, lhand, rhand, lshoulder, rshoulder, larm, rarm, lhip, rhip, lknee, rknee, lwrist, rwrist, lfoot, rfoot, head, pelvis]
    Xsens = [9, 20, 21, 13, 14, 16, 17, 1, 2, 4, 5, 18, 19, 7, 8, 15, 0]

    # DIP_joints = [L_Elbow, R_Elbow, L_Knee, R_Knee, Head, Pelvis]
    DIP_joints = [18, 19, 4, 5, 15, 0]

    DIP_AWO_LFT =[18, 19, 7, 8, 15, 0] 

    # Sensor Configuration Joint Positions
    DIP_spine3 = [18, 19, 4, 5, 9, 0]
    DIP_spine3_elbow = [18, 19, 4, 5, 9, 0]
    DIP_spine3_arm = [18, 19, 16, 17, 4, 5, 9, 0]
    DIP_spine3_arm_only = [18, 19, 16, 17, 9, 0]
    DIP_collar = [18, 19, 4, 5, 13, 14, 0]
    DIP_collar_arm = [18, 19, 16, 17, 4, 5, 13, 14, 0]
    DIP_collar_arm_only = [18, 19, 16, 17, 13, 14, 0]
    DIP_shirt = [18, 19, 16, 17, 4, 5, 13, 14, 9, 3, 0] 
    DIP_shirt_only = [18, 19, 16, 17, 13, 14, 9, 3, 0] 
    dse_complete = [3, 9, 2, 1, 14, 13, 9, 1, 2, 17, 16, 16, 17, 2, 1, 18, 19, 5, 4, 12, 19, 18, 4, 5, 0]


class _VERTEX_CONFIG():
    """ 
    Class that contains vertex configurations for synthesizing.
    Each vertex reflects a point on the SMPL-mesh (NOT SMPLX-mesh)
    that is equippet with a virtual IMU-sensor.
    """
    # Vertex configuration from DIP
    # source: https://meshcapade.wiki/assets/SMPL_body_segmentation/smpl/smpl_vert_segmentation.json
    # DIP_vertices = [leftForeArm, rigthForeArm, leftLeg, rightLeg, head, hips]
    DIP_vertices = [1961, 5424, 1176, 4662, 411, 3021]
    # Xsens = [sternum, lhand, rhand, lshoulder, rshoulder, larm, rarm, lhip, rhip, lknee, rknee, lwrist, rwrist, lfoot, rfoot, head, pelvis]
    Xsens = [3496, 2200, 5662, 707, 5287, 1719, 5188, 958, 4444, 1176, 4662, 1961, 5424, 3365, 6765, 411, 3021] 
    # Custom vertex configurations
    Custom_1_vertices = [1999, 5499, 1199, 4699, 499, 3099, 3021]
    #nodePosition
    AWI_LKF = [1931, 5393, 1176, 4662, 411, 3021]
    AMO_LKF = [1548, 5017, 1176, 4662, 411, 3021]
    AEO_LKF = [1623, 5092, 1176, 4662, 411, 3021]
    AWO_LAO = [1961, 5424, 3190, 6590, 411, 3021]
    AWO_LFT = [1961, 5424, 3365, 6765, 411, 3021]

    # Sensor Configuration Vertex Positions
    DIP_spine3 = [1961, 5424, 1176, 4662, 1305, 3021]
    DIP_spine3_arm = [1961, 5424, 1719, 5188,  1176, 4662, 1305, 3021]
    DIP_spine3_arm_only = [1961, 5424, 1719, 5188, 1305, 3021]
    DIP_spine3_elbow = [1623, 5092, 1176, 4662, 1305, 3021]
    DIP_collar = [1961, 5424, 1176, 4662, 707, 5287, 3021]
    DIP_collar_arm = [1961, 5424, 1719, 5188, 1176, 4662, 707, 5287, 3021]
    DIP_collar_arm_only = [1961, 5424, 1719, 5188, 707, 5287, 3021]
    DIP_shirt = [1961, 5424, 1719, 5188, 1176, 4662, 707, 5287, 1305, 3016, 3021] 
    DIP_shirt_only = [1961, 5424, 1719, 5188, 707, 5287, 1305, 3016, 3021]
    dse_complete = [3016, 3496, 4362, 876, 4197, 707, 1305, 958, 4444, 5335, 1874, 1719, 5188, 4516, 1032, 1623, 5092, 4662, 1177, 411, 5424, 1961, 3322, 6723, 3021]


def _toDictFromClass(myClass):
    attributes = {k: v for k, v in vars(
        myClass).items() if not k.startswith("__")}
    return attributes


def _writeDefaultConfig(path=Constants.config_path):
    """Writes the default config to a .yml file

    Args:
        path (str, optional): Where the config should be saved.
    """
    conf = {}
    conf["paths"] = _toDictFromClass(_Path)
    conf["dataset_paths"] = _toDictFromClass(_DATASET_PATHS)
    conf["dataset_base"] = dataset_base
    conf["preprocessing"] = _toDictFromClass(_Preprocessing)
    conf["TP_joint_set"] = _toDictFromClass(_TP_joint_set)
    conf["joint_config"] = _toDictFromClass(_JOINT_CONFIG)
    conf["vertex_config"] = _toDictFromClass(_VERTEX_CONFIG)

    with open(path, 'w') as file:
        yaml.dump(conf, file)


class Config(object):
    """Collection of all configs for this project.

    Args:
        path (str): Path to the .yml that should be used.
    """

    def __init__(self, path: str):
        super().__init__()
        with open(path, 'r') as file:
            self._conf = yaml.safe_load(file)

    def __getitem__(self, key):
        return self._conf[key]


if __name__ == "__main__":
    path = Constants.config_path
    _writeDefaultConfig(path)
    print("Wrote default config to", path)
