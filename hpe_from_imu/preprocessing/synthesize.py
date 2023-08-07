from codecs import ignore_errors
import glob
import shutil
import os
import pickle

import numpy as np
import torch
from hpe_from_imu.configuration import Config
from hpe_from_imu.configuration import Constants as C
from hpe_from_imu.utils import BodyModel, SMPL_local_to_global
from pytorch3d.transforms.rotation_conversions import (axis_angle_to_matrix,
                                                       matrix_to_axis_angle)
from tqdm import tqdm

"""
Adapted from https://github.com/Xinyu-Yi/TransPose/blob/main/preprocess.py
"""

# vi_mask contains vertices on SMPL mesh where virtual IMUs will placed (used for virtual accelaration)
vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
# ji_mask contains joints which have the same orientation as virtual IMU (used for virtual orientation)
ji_mask = torch.tensor([18, 19, 4, 5, 15, 0])
# Constant for smooting the synthesized accleration
smooth_n = 4

conf = Config(C.config_path)
paths = conf["paths"]
conf_preproc = conf["preprocessing"]


def synthesize_AMASS():
    data_pose, data_trans, data_beta, length = load_AMASS_dataset()

    length = torch.tensor(length, dtype=torch.int)
    shape = torch.tensor(np.asarray(data_beta, np.float32))
    tran = torch.tensor(np.asarray(data_trans, np.float32))
    pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
    pose[:, 23] = pose[:, 37]     # right hand
    pose = pose[:, :24].clone()   # only use body

    # align AMASS global frame with DIP
    amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
    tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
    pose[:, 0] = matrix_to_axis_angle(
        amass_rot.matmul(axis_angle_to_matrix(pose[:, 0])))

    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = synthesize(
        length, shape, tran, pose)

    _save_synthetic_data(out_pose, out_shape, out_tran,
                         out_joint, out_vrot, out_vacc, "AMASS_TP")

def custom_synthesize_AMASS(store_path_key, vertex_key, joint_key, calc_move=False):
    vertex_list = conf["vertex_config"][vertex_key] 
    vertex_mask = torch.tensor(vertex_list)

    joint_list = conf["joint_config"][joint_key] 
    joint_mask = torch.tensor(joint_list)

    data_pose, data_trans, data_beta, length = load_AMASS_dataset()

    length = torch.tensor(length, dtype=torch.int)
    shape = torch.tensor(np.asarray(data_beta, np.float32))
    tran = torch.tensor(np.asarray(data_trans, np.float32))
    pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 52, 3)
    pose[:, 23] = pose[:, 37]     # right hand
    pose = pose[:, :24].clone()   # only use body

    # align AMASS global frame with DIP
    amass_rot = torch.tensor([[[1, 0, 0], [0, 0, 1], [0, -1, 0.]]])
    tran = amass_rot.matmul(tran.unsqueeze(-1)).view_as(tran)
    pose[:, 0] = matrix_to_axis_angle(
        amass_rot.matmul(axis_angle_to_matrix(pose[:, 0])))

    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = custom_synthesize(
        length, shape, tran, pose, vertex_mask, joint_mask)

    # calculate joint movement in m of left and right joints
    if calc_move:
        calc_joint_move(out_joint)

    _save_custom_synthetic_data(out_pose, out_shape, out_tran,
                         out_joint, out_vrot, out_vacc, store_path_key)

def synthesize_DIP():
    data_pose, data_trans, data_beta, length = load_DIP_data()
    length = torch.tensor(length, dtype=torch.int)
    shape = torch.tensor(np.asarray(data_beta, np.float32))
    tran = torch.tensor(np.asarray(data_trans, np.float32))
    pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 24, 3)

    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = synthesize(
        length, shape, tran, pose)

    _save_synthetic_data(out_pose, out_shape, out_tran,
                         out_joint, out_vrot, out_vacc, "DIP_IMU_synth")


def custom_synthesize_DIP(store_path_key, vertex_key, joint_key, calc_move=False, sub=-1, mot=-1,spec=""):
    vertex_list = conf["vertex_config"][vertex_key] 
    vertex_mask = torch.tensor(vertex_list)

    joint_list = conf["joint_config"][joint_key] 
    joint_mask = torch.tensor(joint_list)

    if sub != -1:
        data_pose, data_trans, data_beta, length = load_DIP_sequence(sub, mot, spec)
    else:   
        data_pose, data_trans, data_beta, length = load_DIP_data()

    length = torch.tensor(length, dtype=torch.int)
    shape = torch.tensor(np.asarray(data_beta, np.float32))
    tran = torch.tensor(np.asarray(data_trans, np.float32))
    pose = torch.tensor(np.asarray(data_pose, np.float32)).view(-1, 24, 3)

    # added vertex_mask as input parameter for synthesize task
    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = custom_synthesize(
        length, shape, tran, pose, vertex_mask, joint_mask)
    
    # calculate joint movement in m of left and right joints
    if calc_move:
        calc_joint_move(out_joint)

    # added store_path to specify the folder
    if sub != -1:
        _save_temp_data(out_pose, out_shape, out_tran,
                         out_joint, out_vrot, out_vacc, "s_{:0=2d}{:0=2d}{}".format(sub, mot, spec))
    else:   
        _save_custom_synthetic_data(out_pose, out_shape, out_tran,
                            out_joint, out_vrot, out_vacc, store_path_key)


def calc_joint_move(out_joint):
    print("Calculate joint movement in dataset")
    scene_sum = torch.zeros([24])
    for s in tqdm(range(len(out_joint))):
        for f in range(1, (len(out_joint[s]))):
            joint_dist = torch.zeros([24])
            i = 0
            for (x, y) in zip(out_joint[s][f-1][:], out_joint[s][f][:]):
                d = torch.sqrt(torch.square(y[0]-x[0]) + torch.square(y[1]-x[1]) + torch.square(y[2]-x[2]))
                joint_dist[i]  =  d
                i += 1
            scene_sum = torch.add(scene_sum, joint_dist)

    sum_l_r = []
    sum_l_r.append(scene_sum[[4,7,10,16,18,20,22]].cpu().detach().numpy())
    sum_l_r.append(scene_sum[[5,8,11,17,19,21,23]].cpu().detach().numpy())

    print("{:<20}|".format("Joint Movement"), ("{:<10}" *7).format("Knee", "Ankle", "Foot", "Shoulder", "Ellbow", "Wrist", "Hand"))
    for i, name in enumerate(['Left (m)', 'Right (m)']):
            #print('%s' % (name))
            print("{:<20}|".format(name), ("{:<10.2f}" *7).format(sum_l_r[i][0], sum_l_r[i][1], sum_l_r[i][2], sum_l_r[i][3], sum_l_r[i][4], 
            sum_l_r[i][5], sum_l_r[i][6]))


def load_AMASS_dataset():
    data_pose, data_trans, data_beta, length = [], [], [], []
    for ds_name in conf_preproc["AMASS_dataset"]:
        print('Reading', ds_name)
        for npz_fname in tqdm(glob.glob(os.path.join(paths["AMASS_raw"], ds_name, '*/*_poses.npz'))):
            try:
                cdata = np.load(npz_fname)
            except:
                continue

            framerate = int(cdata['mocap_framerate'])
            if framerate == 120:
                step = 2
            elif framerate == 60 or framerate == 59:
                step = 1
            else:
                continue

            data_pose.extend(cdata['poses'][::step].astype(np.float32))
            data_trans.extend(cdata['trans'][::step].astype(np.float32))
            data_beta.append(cdata['betas'][:10])
            length.append(cdata['poses'][::step].shape[0])

    assert len(data_pose) != 0, 'AMASS dataset not found. Check configuration'
    return data_pose, data_trans, data_beta, length


def load_DIP_data():
    data_pose, data_trans, data_beta, length = [], [], [], []
    print('\rReading DIP-IMU')
    # sort files to split training and test set (done in preprocess.py: preprocess_DIP_TP_custom_synth())
    files = glob.glob(os.path.join(paths["DIP_IMU"], '*/*.pkl'))
    files.sort()
    for pkl_fname in tqdm(files):
        try:
            with open(pkl_fname, 'rb') as f:
                cdata = pickle.load(f, encoding='latin1')
        except:
            continue

        step = 1

        data_pose.extend(cdata['gt'][::step].astype(np.float32))
        data_trans.extend(
            np.zeros((cdata['gt'][::step].shape[0], 3), dtype=np.float32))
        data_beta.append(np.zeros((10), dtype=np.float32))
        length.append(cdata['gt'][::step].shape[0])

    assert len(data_pose) != 0, 'DIP dataset not found. Check configuration'
    return data_pose, data_trans, data_beta, length

def load_DIP_sequence(sub, mot, spec):
    data_pose, data_trans, data_beta, length = [], [], [], []
    print('\rReading DIP-IMU sequence:', "s_{:0=2d}".format(sub), "{:0=2d}{}.pkl".format(mot, spec))
    pkl_fname = os.path.join(paths["DIP_IMU"], "s_{:0=2d}".format(sub), "{:0=2d}{}.pkl".format(mot, spec))
    try:
        with open(pkl_fname, 'rb') as f:
            cdata = pickle.load(f, encoding='latin1')
    except:
        print("failed to load: ", pkl_fname)

    step = 1
    data_pose.extend(cdata['gt'][::step].astype(np.float32))
    data_trans.extend(
        np.zeros((cdata['gt'][::step].shape[0], 3), dtype=np.float32))
    data_beta.append(np.zeros((10), dtype=np.float32))
    length.append(cdata['gt'][::step].shape[0])

    assert len(data_pose) != 0, 'DIP dataset not found. Check configuration'
    return data_pose, data_trans, data_beta, length

def synthesize(length, shape, tran, pose):
    b = 0
    body_model = BodyModel()
    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []
    print('Synthesizing IMU accelerations and orientations')
    for i, l in tqdm(list(enumerate(length))):
        if l <= 12:
            b += l
            print('\tdiscard one sequence with length', l)
            continue
        p = pose[b:b + l]
        joints, vertices = body_model.forward_kinematics(
            pose=p, shape=shape[i], translation=tran[b:b + l])
        out_pose.append(pose[b:b + l].clone())  # N, 24, 3
        out_tran.append(tran[b:b + l].clone())  # N, 3
        out_shape.append(shape[i].clone())  # 10
        out_joint.append(joints[:, :24].contiguous().clone())  # N, 24, 3
        out_vacc.append(_synthesize_accleration(
            vertices[:, vi_mask]))  # N, 6, 3
        out_vrot.append(SMPL_local_to_global(
            p)[:, ji_mask])  # N, 6, 3, 3
        b += l
    return out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc

def custom_synthesize(length, shape, tran, pose, vertex_mask, joint_mask):
    b = 0
    body_model = BodyModel()
    out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc = [], [], [], [], [], []
    print('Synthesizing IMU accelerations and orientations')
    for i, l in tqdm(list(enumerate(length))):
        if l <= 12:
            b += l
            print('\tdiscard one sequence with length', l)
            continue
        p = pose[b:b + l]
        joints, vertices = body_model.forward_kinematics(
            pose=p, shape=shape[i], translation=tran[b:b + l])
        out_pose.append(pose[b:b + l].clone())  # N, 24, 3
        out_tran.append(tran[b:b + l].clone())  # N, 3
        out_shape.append(shape[i].clone())  # 10
        out_joint.append(joints[:, :24].contiguous().clone())  # N, 24, 3
        out_vacc.append(_synthesize_accleration(
            vertices[:, vertex_mask]))  # N, M, 3 with N: seq. length, M: number of joints
        out_vrot.append(axis_angle_to_matrix(SMPL_local_to_global(
            p))[:, joint_mask])  # N, M, 3, 3 with N: seq. length, M: number of joints
        b += l
    return out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc


def _synthesize_accleration(v):
    """
    Synthesize accelerations from vertex positions.
    """
    mid = smooth_n // 2
    # Formula used is backwards second order but offset by one
    # 3600 hardcoded from 60 fps
    acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1])
                       * 3600 for i in range(0, v.shape[0] - 2)])
    acc = torch.cat(
        (torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
    # Smoothing the results
    if mid != 0:
        acc[smooth_n:-smooth_n] = torch.stack(
            [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
             for i in range(0, v.shape[0] - smooth_n * 2)])
    return acc


def _save_synthetic_data(out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc, dataset_name):
    print("Saving")
    os.makedirs(paths[dataset_name], exist_ok=True)
    torch.save(out_joint, paths[dataset_name + "_joints"])
    torch.save(out_pose, paths[dataset_name + "_poses"])
    torch.save(out_shape, paths[dataset_name + "_shapes"])
    torch.save(out_tran, paths[dataset_name + "_trans"])
    torch.save(out_vacc, paths[dataset_name + "_vaccs"])
    torch.save(out_vrot, paths[dataset_name + "_vrots"])
    print(
        f"Synthesized ${dataset_name} dataset saved at ${paths[dataset_name]}")


def _save_custom_synthetic_data(out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc, store_path_key):
    print("Saving")
    # make directory for synthetic dataset
    os.makedirs(paths[store_path_key], exist_ok=True)
    # automatic path generation
    torch.save(out_joint, os.path.join(paths[store_path_key], "joints.pt"))
    torch.save(out_pose, os.path.join(paths[store_path_key], "poses.pt"))
    torch.save(out_shape, os.path.join(paths[store_path_key], "shapes.pt"))
    torch.save(out_tran, os.path.join(paths[store_path_key], "trans.pt"))
    torch.save(out_vacc, os.path.join(paths[store_path_key], "vaccs.pt"))
    torch.save(out_vrot, os.path.join(paths[store_path_key], "vrots.pt"))

    print(
        f"Synthesized ${store_path_key} dataset saved at ${paths[store_path_key]} with customized vertices/joints")

def _save_temp_data(out_pose, out_shape, out_tran, out_joint, out_vrot, out_vacc, name):
    store_path = os.path.join(paths["workspace_dir"], "temp", name)
    os.makedirs(store_path, exist_ok=True)
    print(f"Saving temp data at:", store_path)
    # automatic path generation
    torch.save(out_joint, os.path.join(store_path, "joints.pt"))
    torch.save(out_pose, os.path.join(store_path, "poses.pt"))
    torch.save(out_shape, os.path.join(store_path, "shapes.pt"))
    torch.save(out_tran, os.path.join(store_path, "trans.pt"))
    torch.save(out_vacc, os.path.join(store_path, "vaccs.pt"))
    torch.save(out_vrot, os.path.join(store_path, "vrots.pt"))

    print(
        f"Synthesized ${name} dataset saved at ${store_path}")

def remove_temp_data(name):
    store_path = os.path.join(paths["workspace_dir"], "temp", name)
    shutil.rmtree(store_path)
    print(f"Removed directory:", store_path)


if __name__ == '__main__':
    #synthesize_AMASS()
    #custom_synthesize_AMASS()
    #synthesize_DIP()
    #custom_synthesize_DIP()
    #load_DIP_sequence(10, 2, "")
    #custom_synthesize_DIP("DIP_collar", "DIP_collar", "DIP_collar", calc_move=False, sub=10, mot=2,spec="")
    remove_temp_data("s_1002")
