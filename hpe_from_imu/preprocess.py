import torch
import os
import pickle
from tqdm import tqdm
import numpy as np
from hpe_from_imu.configuration import Config
from hpe_from_imu.configuration import Constants as C
from hpe_from_imu.preprocessing import synthesize_AMASS, synthesize_DIP, custom_synthesize_DIP, custom_synthesize_AMASS

conf = Config(C.config_path)
paths = conf["paths"]
dataset_paths = conf["dataset_paths"]
conf_preprocessing = conf["preprocessing"]
reduced_joint_set = conf["TP_joint_set"]["reduced"]
leaf_joint_set = conf["TP_joint_set"]["leaf"]


def prep_equal_sequences(foldername, filename, sequence_length=300):
    data = torch.load(os.path.join(foldername, filename))
    accs, oris, poses = data["acc"], data["ori"], data["pose"]
    eq_accs, eq_oris, eq_poses = [], [], []
    for i in tqdm(range(len(accs)), desc=filename):
        acc = accs[i]
        ori = oris[i]
        pose = poses[i]
        acc_splits, ori_splits, pose_splits = acc.split(
            sequence_length), ori.split(sequence_length), pose.split(sequence_length)
        for j in range(len(acc_splits) - 1):
            eq_accs.append(acc_splits[j].clone())
            eq_oris.append(ori_splits[j].clone())
            eq_poses.append(pose_splits[j].clone())
        if (len(acc_splits) % sequence_length == 0):
            eq_accs.append(acc_splits[-1].clone())
            eq_oris.append(ori_splits[-1].clone())
            eq_poses.append(pose_splits[-1].clone())
    torch.save({'acc': eq_accs, 'ori': eq_oris, 'pose': eq_poses},
               os.path.join(foldername, str(sequence_length) + "-" + filename))


def prep_equal_sequences_with_joints(foldername, filename, sequence_length=300):
    data = torch.load(os.path.join(foldername, filename))
    accs, oris, poses, joints = data["acc"], data["ori"], data["pose"], data["joint"]
    eq_accs, eq_oris, eq_poses, eq_joints = [], [], [], []
    for i in tqdm(range(len(accs)), desc=filename):
        acc = accs[i]
        ori = oris[i]
        pose = poses[i]
        joint = joints[i]
        acc_splits, ori_splits, pose_splits, joint_splits = acc.split(sequence_length), ori.split(
            sequence_length), pose.split(sequence_length), joint.split(sequence_length)
        for j in range(len(acc_splits) - 1):
            eq_accs.append(acc_splits[j].clone())
            eq_oris.append(ori_splits[j].clone())
            eq_poses.append(pose_splits[j].clone())
            eq_joints.append(joint_splits[j].clone())
        if (len(acc_splits) % sequence_length == 0):
            eq_accs.append(acc_splits[-1].clone())
            eq_oris.append(ori_splits[-1].clone())
            eq_poses.append(pose_splits[-1].clone())
            eq_joints.append(joint_splits[-1].clone())
    torch.save({'acc': eq_accs, 'ori': eq_oris, 'pose': eq_poses, 'joint': eq_joints},
               os.path.join(foldername, str(sequence_length) + "-" + filename))


def preprocess_TotalCapture_DIP():
    accs, oris, poses,  = [], [], []
    for file in sorted(os.listdir(paths["TC_DIP"])):
        data = pickle.load(
            open(os.path.join(paths["TC_DIP"], file), 'rb'), encoding='latin1')
        ori = torch.from_numpy(data['ori']).float()[
            :, torch.tensor([2, 3, 0, 1, 4, 5])]
        acc = torch.from_numpy(data['acc']).float()[
            :, torch.tensor([2, 3, 0, 1, 4, 5])]
        pose = torch.from_numpy(data['gt']).float().view(-1, 24, 3)

        # acc/ori and gt pose do not match in the dataset
        if acc.shape[0] < pose.shape[0]:
            pose = pose[:acc.shape[0]]
        elif acc.shape[0] > pose.shape[0]:
            acc = acc[:pose.shape[0]]
            ori = ori[:pose.shape[0]]

        assert acc.shape[0] == ori.shape[0] and ori.shape[0] == pose.shape[0]
        accs.append(acc)    # N, 6, 3
        oris.append(ori)    # N, 6, 3, 3
        poses.append(pose)  # N, 24, 3

    os.makedirs(paths["TC_DIP_preprocessed"], exist_ok=True)
    torch.save({'acc': accs, 'ori': oris, 'pose': poses},
               os.path.join(paths["TC_DIP_preprocessed"], 'test.pt'))
    print('Preprocessed TotalCapture dataset is saved at',
          paths["TC_DIP_preprocessed"])
    prep_TC_DIP_equal_sequences()


def prep_TC_DIP_equal_sequences(sequence_length=300):
    print("Preparing Total Capture (from DIP) sequences of equal length with length", sequence_length)
    prep_equal_sequences(
        paths["TC_DIP_preprocessed"], "test.pt", sequence_length)


def preprocess_AMASS_TP():
    # Make sure you synthesized the dataset beforehand
    print("Preprocessing Synthetic AMASS dataset from TP authors.")
    acc = torch.load(paths["AMASS_TP_vaccs"])
    ori = torch.load(paths["AMASS_TP_vrots"])
    pose = torch.load(paths["AMASS_TP_poses"])

    os.makedirs(paths["AMASS_TP_preprocessed"], exist_ok=True)
    torch.save({'acc': acc, 'ori': ori, 'pose': pose},
               os.path.join(paths["AMASS_TP_preprocessed"], 'train.pt'))
    print('Preprocessed AMASS (from TP) dataset is saved at',
          paths["AMASS_TP_preprocessed"])
    prep_AMASS_TP_equal_sequences()


def prep_AMASS_TP_equal_sequences(sequence_length=300):
    print("Preparing AMASS (from TP) sequences of equal length with length", sequence_length)
    prep_equal_sequences(
        paths["AMASS_TP_preprocessed"], "train.pt", sequence_length)

def preprocess_AMASS_TP_custom_synth(raw_path_key, work_path_key):
    """_summary_
    Use to preprocess raw customized synthetic AMASS data to a working dataset.
    It is designed for DIP-like networks.

    Args:
        raw_path_key (_type_): _description_
        work_path_key (_type_): _description_
    """
    # Make sure you synthesized the dataset beforehand
    print("Preprocessing Customized AMASS dataset from TP authors.")
    acc = torch.load(os.path.join(paths[raw_path_key], "vaccs.pt"))
    ori = torch.load(os.path.join(paths[raw_path_key], "vrots.pt"))
    pose = torch.load(os.path.join(paths[raw_path_key], "poses.pt"))

    os.makedirs(paths[work_path_key], exist_ok=True)
    torch.save({'acc': acc, 'ori': ori, 'pose': pose},
               os.path.join(paths[work_path_key], 'train.pt'))
    print('Preprocessed AMASS (from TP) dataset is saved at',
          paths[work_path_key])
    prep_AMASS_TP_custom_synth_equal_sequences(work_path_key)


def prep_AMASS_TP_custom_synth_equal_sequences(work_path_key, sequence_length=300):
    print("Preparing AMASS (from TP) sequences of equal length with length", sequence_length)
    prep_equal_sequences(
        paths[work_path_key], "train.pt", sequence_length)


def preprocess_AMASS_TP_training():
    # Make sure you synthesized the dataset beforehand
    print("Preprocessing Synthetic AMASS dataset from TP authors for TP training")
    acc = torch.load(paths["AMASS_TP_vaccs"])
    ori = torch.load(paths["AMASS_TP_vrots"])
    pose = torch.load(paths["AMASS_TP_poses"])
    joint = torch.load(paths["AMASS_TP_joints"])

    os.makedirs(paths["TP_training"], exist_ok=True)
    torch.save({'acc': acc, 'ori': ori, 'pose': pose, 'joint': joint},
               os.path.join(paths["TP_training"], 'train.pt'))
    print('Preprocessed AMASS (from TP) dataset is saved at',
          paths["TP_training"])
    prep_AMASS_TP_equal_sequences_TP_training()


def prep_AMASS_TP_equal_sequences_TP_training(sequence_length=300):
    print("Preparing AMASS (from TP) for TP training sequences of equal length with length", sequence_length)
    prep_equal_sequences_with_joints(
        paths["TP_training"], "train.pt", sequence_length)


def preprocess_TP_training():
    # Make sure you synthesized the dataset beforehand

    # preprocess_AMASS_TP_training()

    print("Preprocessing Synthetic AMASS dataset from TP authors for training of S1, S2 and S3 of TransPose model")

    data = torch.load(os.path.join(dataset_paths["TP_training"]))

    acc = torch.stack(data["acc"]).flatten(start_dim=2)
    ori = torch.stack(data["ori"]).flatten(start_dim=2)
    joint = torch.stack(data["joint"])[:, :, 1:]
    pose = torch.stack(data["pose"])

    imu = torch.cat((acc, ori), dim=2)
    s1_out = joint[:, :, leaf_joint_set].flatten(start_dim=2)
    s2_in = torch.cat((s1_out, imu), dim=2)
    s3_in = torch.cat((joint.flatten(start_dim=2), imu), dim=2)
    s3_out = pose[:, :, reduced_joint_set].flatten(start_dim=2)

    os.makedirs(paths["TP_training"], exist_ok=True)
    os.makedirs(paths["TP_training_S1"], exist_ok=True)
    os.makedirs(paths["TP_training_S2"], exist_ok=True)
    os.makedirs(paths["TP_training_S3"], exist_ok=True)
    os.makedirs(paths["TP_training_total"], exist_ok=True)

    torch.save({'input': imu, 'output': s1_out}, os.path.join(
        paths["TP_training_S1"], '300-train.pt'))
    torch.save({'input': s2_in, 'output': joint.flatten(start_dim=2)}, os.path.join(
        paths["TP_training_S2"], '300-train.pt'))
    torch.save({'input': s3_in, 'output': s3_out}, os.path.join(
        paths["TP_training_S3"], '300-train.pt'))
    torch.save({'acc': acc, 'ori': ori,  'pose': s3_out}, os.path.join(
        paths["TP_training_total"], '300-train.pt'))
    print('Preprocessed AMASS (from TP) dataset for TP training is saved at',
          paths["TP_training"])


def preprocess_AMASS_DIP():
    """_summary_
    Use to process raw Synthetic_60FPS AMASS dataset to working AMASS dataset in DIPnet format
    """
    print("Preprocessing Synthetic AMASS dataset from DIP authors")

    test_split = conf_preprocessing["AMASS_test_split"]
    test_accs, test_oris, test_poses = [], [], []
    train_accs, train_oris, train_poses = [], [], []

    for sub_dataset in tqdm(os.listdir(paths["AMASS_DIP"]), desc="All datasets", position=1):
        if os.path.isfile(os.path.join(paths["AMASS_DIP"], sub_dataset)):
            continue
        for motion_name in tqdm(os.listdir(os.path.join(paths["AMASS_DIP"], sub_dataset)), desc=sub_dataset, position=0):
            path = os.path.join(
                paths["AMASS_DIP"], sub_dataset, motion_name)
            with open(path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            acc = torch.from_numpy(np.asarray(data['acc'])).float()
            ori = torch.from_numpy(np.asarray(data['ori'])).float()
            pose = torch.from_numpy(np.asarray(data['poses'])).float()

            # Count NaN values
            acc_nan_count = torch.nonzero(
                torch.isnan(torch.flatten(acc))).size(dim=0)
            ori_nan_count = torch.nonzero(
                torch.isnan(torch.flatten(ori))).size(dim=0)
            if ori_nan_count != 0 or acc_nan_count != 0:
                print(path, "contains %s NaN in acc and %s NaN in ori" %
                      (acc_nan_count, ori_nan_count))

            if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                if sub_dataset in test_split:
                    test_accs.append(acc.clone())
                    test_oris.append(ori.clone())
                    test_poses.append(pose.clone())
                else:
                    train_accs.append(acc.clone())
                    train_oris.append(ori.clone())
                    train_poses.append(pose.clone())
            else:
                print('AMASS: %s has too much nan! Discard!' % (path))

    os.makedirs(paths["AMASS_DIP_preprocessed"], exist_ok=True)
    torch.save({'acc': test_accs, 'ori': test_oris, 'pose': test_poses},
               os.path.join(paths["AMASS_DIP_preprocessed"], 'test.pt'))
    torch.save({'acc': train_accs, 'ori': train_oris, 'pose': train_poses},
               os.path.join(paths["AMASS_DIP_preprocessed"], 'train.pt'))
    print('Preprocessed AMASS (from DIP) dataset is saved at',
          paths["AMASS_DIP_preprocessed"])
    prep_AMASS_DIP_equal_sequences()


def prep_AMASS_DIP_equal_sequences(sequence_length=300):
    print("Preparing AMASS (from DIP) sequences of equal length with length", sequence_length)
    prep_equal_sequences(
        paths["AMASS_DIP_preprocessed"], "test.pt", sequence_length)
    prep_equal_sequences(
        paths["AMASS_DIP_preprocessed"], "train.pt", sequence_length)


def prep_AMASS_DIP_example(sub_dataset, motion_name, seperate_files=False):
    path = os.path.join(
        paths["AMASS_DIP"], sub_dataset, motion_name)
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    acc = torch.from_numpy(np.asarray(data['acc'])).float()
    ori = torch.from_numpy(np.asarray(data['ori'])).float()
    pose = torch.from_numpy(np.asarray(data['poses'])).float()
    os.makedirs(paths["example_dir"], exist_ok=True)
    if (seperate_files):
        torch.save(acc, os.path.join(
            paths["example_dir"], sub_dataset + "-" + motion_name + '-acc.pt'))
        torch.save(ori, os.path.join(
            paths["example_dir"], sub_dataset + "-" + motion_name + '-ori.pt'))
        torch.save(pose, os.path.join(
            paths["example_dir"], sub_dataset + "-" + motion_name + '-gt-pose.pt'))
    else:
        torch.save({'acc': [acc], 'ori': [ori], 'pose': [pose]}, os.path.join(
            paths["example_dir"], sub_dataset + "-" + motion_name + '-example.pt'))
    print('Generated example. Saved at', paths["example_dir"])


def preprocess_DIP_TP_synth():
    # Make sure you synthesized the dataset beforehand
    print("Preprocessing Synthetic DIP-IMU dataset")
    acc = torch.load(paths["DIP_IMU_synth_vaccs"])
    ori = torch.load(paths["DIP_IMU_synth_vrots"])
    pose = torch.load(paths["DIP_IMU_synth_poses"])

    os.makedirs(paths["DIP_IMU_synth_preprocessed"], exist_ok=True)
    torch.save({'acc': acc, 'ori': ori, 'pose': pose},
               os.path.join(paths["DIP_IMU_synth_preprocessed"], 'train.pt'))
    print('Preprocessed DIP-IMU (synth by TP) dataset is saved at',
          paths["DIP_IMU_synth_preprocessed"])
    prep_DIP_TP_synth_equal_sequences()


def prep_DIP_TP_synth_equal_sequences(sequence_length=300):
    print("Preparing Synthetic DIP-IMU sequences of equal length with length", sequence_length)
    prep_equal_sequences(
        paths["DIP_IMU_synth_preprocessed"], "train.pt", sequence_length)

def preprocess_DIP_TP_custom_synth(raw_path_key, work_path_key):
    """_summary_
    Use to preprocess raw customized synthetic DIP-IMU data to a working dataset.
    It is designed for DIP-like networks.

    Args:
        raw_path_key (_type_): _description_
        work_path_key (_type_): _description_
    """
    # Make sure you synthesized the dataset beforehand
    print("Preprocessing Customized DIP-IMU dataset")
    acc = torch.load(os.path.join(paths[raw_path_key], "vaccs.pt"))
    ori = torch.load(os.path.join(paths[raw_path_key], "vrots.pt"))
    pose = torch.load(os.path.join(paths[raw_path_key], "poses.pt"))

    os.makedirs(paths[work_path_key], exist_ok=True)
    # [:42] takes the subjects 1-8 for use in training set and [42:] subjects 9-10 for the test set
    torch.save({'acc': acc[:42], 'ori': ori[:42], 'pose': pose[:42]},
               os.path.join(paths[work_path_key], 'train.pt'))
    torch.save({'acc': acc[42:], 'ori': ori[42:], 'pose': pose[42:]},
               os.path.join(paths[work_path_key], 'test.pt'))
    print('Preprocessed DIP-IMU (synth by TP) dataset is saved at',
          paths[work_path_key])
    prep_DIP_TP_custom_synth_equal_sequences(work_path_key)

def preprocess_DIP_TP_from_temp(name):
    """_summary_
    Use to preprocess raw customized synthetic DIP-IMU data in temp folder.
    It is designed for DIP-like networks.

    Args:
        name (string): _description_
    """
    store_path = os.path.join(paths["workspace_dir"], "temp", name)
    # Make sure you synthesized the dataset beforehand
    print("Preprocessing Customized DIP-IMU dataset")
    acc = torch.load(os.path.join(store_path, "vaccs.pt"))
    ori = torch.load(os.path.join(store_path, "vrots.pt"))
    pose = torch.load(os.path.join(store_path, "poses.pt"))

    torch.save({'acc': acc, 'ori': ori, 'pose': pose},
               os.path.join(store_path, 'train.pt'))
    print('Preprocessed DIP-IMU (synth by TP) dataset is saved at', store_path)
    prep_DIP_TP_from_temp_equal_sequences(store_path)

def prep_DIP_TP_from_temp_equal_sequences(store_path, sequence_length=300):
    print("Preparing Synthetic DIP-IMU sequences of equal length with length", sequence_length)
    prep_equal_sequences(store_path, "train.pt", sequence_length)

def prep_DIP_TP_custom_synth_equal_sequences(work_path_key, sequence_length=300):
    print("Preparing Synthetic DIP-IMU sequences of equal length with length", sequence_length)
    prep_equal_sequences(
        paths[work_path_key], "train.pt", sequence_length)
    prep_equal_sequences(
        paths[work_path_key], "test.pt", sequence_length)    


def preprocess_DIP_IMU(mask: str = "DIP_IMU_mask", data_work: str = "DIP_IMU_preprocessed", validation=False):
    """_summary_
    Use to get working dataset of original DIP node configuration.

    Funktion takes raw data from /home/.../data/dataset_raw/DIP_IMU_and_Others/DIP_IMU
    and loads every sequence and motion from dataset, reads acceleration, orientation 
    and pose for masked joints to fills nan with nearest neighbors.
    All sequences are the added to two large sequences, one test and one train sequence.
    The test and train sequences are than stored at /home/.../data/dataset_work/DIP_IMU
    in the subfolders train.pt and test.pt
    Following a separation in sequences of 300 frames is done.
    """
    print("Preprocessing DIP-IMU dataset")

    imu_mask = conf_preprocessing[mask]
    test_split = conf_preprocessing["DIP_test_split"]
    valitation_split = conf_preprocessing["DIP_validation_split"]
    test_accs, test_oris, test_poses = [], [], []
    train_accs, train_oris, train_poses = [], [], []
    validation_accs, validation_oris, validation_poses = [], [], []

    for subject_name in tqdm(os.listdir(paths["DIP_IMU"]), desc="All subjects", position=1):
        for motion_name in tqdm(os.listdir(os.path.join(
                paths["DIP_IMU"], subject_name)), desc=subject_name, position=0):

            path = os.path.join(
                paths["DIP_IMU"], subject_name, motion_name)
            with open(path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
            ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
            pose = torch.from_numpy(data['gt']).float()

            # fill nan with nearest neighbors
            for _ in range(4):
                acc[1:].masked_scatter_(torch.isnan(
                    acc[1:]), acc[:-1][torch.isnan(acc[1:])])
                ori[1:].masked_scatter_(torch.isnan(
                    ori[1:]), ori[:-1][torch.isnan(ori[1:])])
                acc[:-1].masked_scatter_(torch.isnan(acc[:-1]),
                                         acc[1:][torch.isnan(acc[:-1])])
                ori[:-1].masked_scatter_(torch.isnan(ori[:-1]),
                                         ori[1:][torch.isnan(ori[:-1])])

            acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]
            if torch.isnan(acc).sum() == 0 and torch.isnan(ori).sum() == 0 and torch.isnan(pose).sum() == 0:
                if subject_name in test_split:
                    test_accs.append(acc.clone())
                    test_oris.append(ori.clone())
                    test_poses.append(pose.clone())
                
                elif validation and (subject_name+motion_name) in valitation_split:
                    validation_accs.append(acc.clone())
                    validation_oris.append(ori.clone())
                    validation_poses.append(pose.clone())

                else:
                    train_accs.append(acc.clone())
                    train_oris.append(ori.clone())
                    train_poses.append(pose.clone())
            else:
                pass
                # print('DIP-IMU: %s has too much nan! Discard!' % (path))

    os.makedirs(paths[data_work], exist_ok=True)
    torch.save({'acc': test_accs, 'ori': test_oris, 'pose': test_poses},
               os.path.join(paths[data_work], 'test.pt'))
    torch.save({'acc': train_accs, 'ori': train_oris, 'pose': train_poses},
               os.path.join(paths[data_work], 'train.pt'))
    torch.save({'acc': validation_accs, 'ori': validation_oris, 'pose': validation_poses},
               os.path.join(paths[data_work], 'validation.pt'))
    print('Preprocessed DIP-IMU dataset is saved at',
          paths[data_work])

    prep_DIP_IMU_equal_sequences(data_work=data_work, validation=validation)


def prep_DIP_IMU_equal_sequences(sequence_length=300, data_work: str = "DIP_IMU_preprocessed", validation=False):
    print("Preparing DIP sequences of equal length with length", sequence_length)
    prep_equal_sequences(paths[data_work], "test.pt", sequence_length)
    prep_equal_sequences(paths[data_work], "train.pt", sequence_length)
    if validation:
        prep_equal_sequences(paths[data_work], "validation.pt", sequence_length)


def prep_DIP_example(subject_name, motion_name, seperate_files=False, mask: str = "DIP_IMU_mask"):
    imu_mask = conf_preprocessing[mask]
    path = os.path.join(
        paths["DIP_IMU"], subject_name, motion_name + ".pkl")
    data = pickle.load(open(path, 'rb'), encoding='latin1')
    acc = torch.from_numpy(data['imu_acc'][:, imu_mask]).float()
    ori = torch.from_numpy(data['imu_ori'][:, imu_mask]).float()
    pose = torch.from_numpy(data['gt']).float()

    for _ in range(4):
        acc[1:].masked_scatter_(torch.isnan(
            acc[1:]), acc[:-1][torch.isnan(acc[1:])])
        ori[1:].masked_scatter_(torch.isnan(
            ori[1:]), ori[:-1][torch.isnan(ori[1:])])
        acc[:-1].masked_scatter_(torch.isnan(acc[:-1]),
                                 acc[1:][torch.isnan(acc[:-1])])
        ori[:-1].masked_scatter_(torch.isnan(ori[:-1]),
                                 ori[1:][torch.isnan(ori[:-1])])

    acc, ori, pose = acc[6:-6], ori[6:-6], pose[6:-6]

    os.makedirs(paths["example_dir"], exist_ok=True)
    if (seperate_files):
        torch.save(acc, os.path.join(
            paths["example_dir"], subject_name + "-" + motion_name + '-acc.pt'))
        torch.save(ori, os.path.join(
            paths["example_dir"], subject_name + "-" + motion_name + '-ori.pt'))
        torch.save(pose.view(-1, 24, 3), os.path.join(
            paths["example_dir"], subject_name + "-" + motion_name + '-gt-pose.pt'))
    else:
        torch.save({'acc': [acc], 'ori': [ori], 'pose': [pose]}, os.path.join(
            paths["example_dir"], subject_name + "-" + motion_name + '-example.pt'))
    print('Generated example. Saved at', paths["example_dir"])


if __name__ == '__main__':
    # DIP-raw to DIP-work with original configuration
    # preprocess_DIP_IMU()
    #preprocess_DIP_IMU(mask = "DIP_IMU_mask", data_work = "DIP_IMU_REF_preprocessed")

    # Synthetic_60FPS to AMASS-DIP working dataset with original configuration
    # preprocess_AMASS_DIP()

    # AMASS-raw to AMASS-raw-customized
    #custom_synthesize_AMASS("Custom_AMASS_path", "Custom_1_vertices", "Custom_1_joints")
    # AMASS-raw-customized to AMASS-work-customized
    #preprocess_AMASS_TP_custom_synth("Custom_AMASS_path", "Custom_AMASS_path_preprocessed")
    # synthesize_AMASS()
    # preprocess_AMASS_TP()
    
    # DIP-raw to DIP-raw-customized
    #custom_synthesize_DIP("Custom_DIP_path", "Custom_1_vertices", "Custom_1_joints")
    # DIP-raw-customized to DIP-work_customized
    #preprocess_DIP_TP_custom_synth("Custom_DIP_path", "Custom_DIP_path_preprocessed")
    # synthesize_DIP()
    # preprocess_DIP_TP_synth()

    # preprocess_TotalCapture_DIP()
    # preprocess_TP_training()

    prep_DIP_example("s_10", "05")
    # prep_AMASS_DIP_example("AMASS_ACCAD", "s011walkdog_dynamics.pkl")

    # node Positions
    # Reference
    #custom_synthesize_AMASS("Custom_AMASS_path", "DIP_vertices", "DIP_joints")
    #preprocess_AMASS_TP_custom_synth("Custom_AMASS_path", "Custom_AMASS_path_preprocessed")
    #custom_synthesize_DIP("DIP_REF", "DIP_vertices", "DIP_joints")#, calc_move=True)
    #preprocess_DIP_TP_custom_synth("DIP_REF", "DIP_REF_preprocessed")

    # Node Variations
    #custom_synthesize_AMASS("AMASS_AWO_LFT", "AWO_LFT", "DIP_AWO_LFT")
    #preprocess_AMASS_TP_custom_synth("AMASS_AWO_LFT", "AMASS_AWO_LFT_preprocessed")
    #custom_synthesize_DIP("DIP_AWI_LKF", "AWI_LKF", "DIP_joints")
    #preprocess_DIP_TP_custom_synth("DIP_AWI_LKF", "DIP_AWI_LKF_preprocessed")

    # Sensor Configuration
    #custom_synthesize_AMASS("AMASS_spine3", "DIP_spine3", "DIP_spine3")
    #preprocess_AMASS_TP_custom_synth("AMASS_spine3", "AMASS_spine3_preprocessed")
    #custom_synthesize_DIP("DIP_spine3", "DIP_spine3", "DIP_spine3")
    #preprocess_DIP_TP_custom_synth("DIP_spine3", "DIP_spine3_preprocessed")
    
    custom_synthesize_AMASS("AMASS_dse_complete", "dse_complete", "dse_complete")
    preprocess_AMASS_TP_custom_synth("AMASS_dse_complete", "AMASS_dse_complete_preprocessed")
    custom_synthesize_DIP("DIP_dse_complete", "dse_complete", "dse_complete")
    preprocess_DIP_TP_custom_synth("DIP_dse_complete", "DIP_dse_complete_preprocessed")

    #custom_synthesize_AMASS("AMASS_spine3_arm_only", "DIP_spine3_arm_only", "DIP_spine3_arm_only")
    #preprocess_AMASS_TP_custom_synth("AMASS_spine3_arm_only", "AMASS_spine3_arm_only_preprocessed")
    #custom_synthesize_DIP("DIP_spine3_arm_only", "DIP_spine3_arm_only", "DIP_spine3_arm_only")
    #preprocess_DIP_TP_custom_synth("DIP_spine3_arm_only", "DIP_spine3_arm_only_preprocessed")

    #custom_synthesize_AMASS("AMASS_spine3_ellbow", "DIP_spine3_ellbow", "DIP_spine3")
    #preprocess_AMASS_TP_custom_synth("AMASS_spine3_ellbow", "AMASS_spine3_ellbow_preprocessed")
    #custom_synthesize_DIP("DIP_spine3_ellbow", "DIP_spine3_ellbow", "DIP_spine3")
    #preprocess_DIP_TP_custom_synth("DIP_spine3_ellbow", "DIP_spine3_ellbow_preprocessed")

    #custom_synthesize_AMASS("AMASS_collar", "DIP_collar", "DIP_collar")
    #preprocess_AMASS_TP_custom_synth("AMASS_collar", "AMASS_collar_preprocessed")
    #custom_synthesize_DIP("DIP_collar", "DIP_collar", "DIP_collar")
    #preprocess_DIP_TP_custom_synth("DIP_collar", "DIP_collar_preprocessed")

    #custom_synthesize_AMASS("AMASS_collar_arm", "DIP_collar_arm", "DIP_collar_arm")
    #preprocess_AMASS_TP_custom_synth("AMASS_collar_arm", "AMASS_collar_arm_preprocessed")
    #custom_synthesize_DIP("DIP_collar_arm", "DIP_collar_arm", "DIP_collar_arm")
    #preprocess_DIP_TP_custom_synth("DIP_collar_arm", "DIP_collar_arm_preprocessed")

    #custom_synthesize_AMASS("AMASS_collar_arm_only", "DIP_collar_arm_only", "DIP_collar_arm_only")
    #preprocess_AMASS_TP_custom_synth("AMASS_collar_arm_only", "AMASS_collar_arm_only_preprocessed")
    #custom_synthesize_DIP("DIP_collar_arm_only", "DIP_collar_arm_only", "DIP_collar_arm_only")
    #preprocess_DIP_TP_custom_synth("DIP_collar_arm_only", "DIP_collar_arm_only_preprocessed")

    #custom_synthesize_AMASS("AMASS_shirt", "DIP_shirt", "DIP_shirt")
    #preprocess_AMASS_TP_custom_synth("AMASS_shirt", "AMASS_shirt_preprocessed")
    #custom_synthesize_DIP("DIP_shirt", "DIP_shirt", "DIP_shirt")
    #preprocess_DIP_TP_custom_synth("DIP_shirt", "DIP_shirt_preprocessed")

    #custom_synthesize_AMASS("AMASS_shirt_only", "DIP_shirt_only", "DIP_shirt_only")
    #preprocess_AMASS_TP_custom_synth("AMASS_shirt_only", "AMASS_shirt_only_preprocessed")
    #custom_synthesize_DIP("DIP_shirt_only", "DIP_shirt_only", "DIP_shirt_only")
    #preprocess_DIP_TP_custom_synth("DIP_shirt_only", "DIP_shirt_only_preprocessed")
