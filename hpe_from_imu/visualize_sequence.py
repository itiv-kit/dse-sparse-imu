import argparse
import os
from datetime import datetime
from os.path import join as join_path
from PIL import Image, ImageFont, ImageDraw
from pyquaternion import Quaternion
from pytorch3d.transforms import matrix_to_axis_angle
from tqdm import tqdm
import numpy as np
import open3d as o3d
import smplx
import torch
import cv2
from matplotlib import cm
from hpe_from_imu.configuration import Config
from hpe_from_imu.configuration import Constants as C

# Color blind palette after Wong et al.
colors = [
    # [0., 0., 0.],
    [0.90196078, 0.62352941, 0.],
    [0.3372549, 0.70588235, 0.91372549],
    [0., 0.61960784, 0.45098039],
    [0.94117647, 0.89411765, 0.25882353],
    [0., 0.44705882, 0.69803922],
    [0.83529412, 0.36862745, 0.],
    [0.8, 0.4745098, 0.65490196]
]


def _text_3d(text, pos=[0, 0, 0], direction=None, degree=-90.0, density=10, font='DejaVuSans.ttf', font_size=16):
    """
    Generate a 3D text point cloud used for visualization. Taken from https://github.com/isl-org/Open3D/issues/2
    :param text: Content of the text
    :param pos: 3D xyz position of the text upper left corner
    :param direction: 3D normalized direction of where the text faces
    :param degree: In plane rotation of text
    :param density: Point cloud density of the text
    :param font: Name of the font - change it according to your system
    :param font_size: Size of the font
    :return: o3d.geoemtry.PointCloud object
    """
    if direction is None:
        direction = (0., 0., 1.)

    font_obj = ImageFont.truetype(font, font_size * density)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(
        img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 100.0 / density)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd


def _setupArgs() -> argparse.ArgumentParser:
    """Setups all the arguments and parses them.

    Returns:
        argparse.ArgumentParser: The parsed argument parser for this script.
    """
    p = argparse.ArgumentParser(description='Visualize SMPL sequence')
    p.add_argument("-j", "--joints",
                   default=False,
                   dest='show_joints',
                   action='store_true',
                   help='Only show joints of models')
    p.add_argument("-c", "--capture",
                   default=False,
                   dest='capture',
                   action='store_true',
                   help='Capture every frame as screen shot and save in capture directory with .avi video-file')
    p.add_argument('-s', '--shape',
                   default=torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                   dest='shape',
                   help='SMPL shape tensor of shape torch.Size([10])',
                   type=torch.Tensor)
    p.add_argument('-f', '--files',
                   dest='files',
                   help='Name of the SMPL sequences that should be visualized with shape seq_len * 24 * 3 or seq_len * 24 * 3 * 3 if parameter ',
                   type=str,
                   nargs='+',
                   required=True)
    p.add_argument('-r', '--rot-mat',
                   dest='rot_mat',
                   help='Should be set to 1 for each SMPL sequence that is not present in euler angels but rotation matrices, otherwise 0',
                   type=int,
                   nargs='*')
    p.add_argument("-v", "--verbose",
                   default=False,
                   dest='verbose',
                   action='store_true',
                   help='Detailed output in terminal')
    p.add_argument("-l", "--show-labels",
                   default=False,
                   dest='show_labels',
                   action='store_true',
                   help='Shows file names above models as label')
    p.add_argument("-e", "--show-errors",
                   default=False,
                   dest='show_errors',
                   action='store_true',
                   help='Shows single vertex error on SMPL-mesh')
    p.add_argument('-k', '--sensor_keys',
                   default="",  
                   dest='sensor_key',
                   help='Name of the SMPL vertex-key to visualize sensors ',
                   type=str,
                   nargs='*')

    return p.parse_args()


def _getArgs() -> tuple[torch.Tensor, list[str], list[bool], bool, bool]:
    """Parses all args for this script and performs a few checks and corrections.

    Returns:
        (Tensor(), list[str], list[bool], bool, bool): Tuple containing the shape, list of files, list of bools whether the files contain rotation matrices, capture flag, verbosity flag and joints flag
    """
    args = _setupArgs()
    files = args.files
    is_rot_mat = args.rot_mat
    if is_rot_mat is None:
        is_rot_mat = [False] * len(files)
    else:
        assert(len(is_rot_mat) is len(files))
        is_rot_mat = [bool(x) for x in is_rot_mat]
    
    return args.shape, files, is_rot_mat, args.capture, args.verbose, args.show_joints, args.show_labels, args.show_errors, args.sensor_key


def _loadPoseAndOri(filename: str, isRotationMatrix: bool, verbose: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads pose and global orientation from the given file and converts poses in rotation matrices form to euler angels if isRotationMatrix is True

    Args:
        filename (str): File to load. If input is "example" the file "example-pose.pt" is going to be loaded from the example directory.
        isRotationMatrix (bool): Whether the corresponding file is in rotation matrix form or not
        verbose (bool): Increases verbosity if true

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Returns tuple of global orientation tensor (seq_len * 1 * 3) and tensor for the SMPL pose parameters (seq_len * 23 * 3)
    """
    path = join_path(paths["example_dir"], filename + "-pose.pt")
    seq = torch.load(path).squeeze()
    if (verbose):
        print("Loaded file %s with sequence of shape %s" %
              (filename, seq.shape))
    if isRotationMatrix:
        seq = matrix_to_axis_angle(seq)
    return torch.split(seq, [1, 23], 1)


def _loadPosesAndOris(files: list[str], isRotationMatrix: list[bool], verbose: bool) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Loads poses and orientations from file list and converts poses in rotation matrices form to euler angels.

    Args:
        files (list[str]): List of files to load. If input is "example" the file "example-pose.pt" is going to be loaded from the example directory. 
        isRotationMatrix (list[bool]): List of whether the corresponding file is in rotation matrix form or not
        verbose (bool): Increases verbosity if true

    Returns:
        tuple[list[Tensor], list[Tensor]]: Returns tuple of two list. The first list contains the global orientation tensors (seq_len * 1 * 3) and the second tensors for the SMPL pose parameters (seq_len * 23 * 3)
    """
    global_oris, poses = ([], [])
    for i, file in enumerate(files):
        cur_ori, cur_pose = _loadPoseAndOri(file, isRotationMatrix[i], verbose)
        global_oris.append(cur_ori)
        poses.append(cur_pose)
    return global_oris, poses


def _getJointsAndVertices(model: smplx.SMPL, shape: torch.Tensor, global_oris: torch.Tensor, poses: torch.Tensor):
    """Calculates the joints and vertices from the given poses for given model, shape and global orientation

    Args:
        model (smplx.SMPL): Used SMPL model or other compatible models
        shape (list[Tensor]): List of SMPL shape tensors. Has to correspond to shape parameter required by the given model. E.g. with simple SMPL model shape of each tensor (10)
        global_oris (list[Tensor]): List of global orientations (seq_len * 1 * 3)
        poses (list[Tensor]): List of SMPL pose parameters (seq_len * 23 * 3)

    Returns:
        list[vertices]: List of vertices for the respective SMPL poses
        list[joints]: List of joint for the respective SMPL poses
    """
    joints = []
    vertices = []
    for i in range(len(global_oris)):
        output = model(body_pose=poses[i], global_orient=global_oris[i],
                       betas=torch.vstack([shape] * len(poses[i])))
        vertices.append(output.vertices.detach().cpu().numpy().squeeze())
        joints.append(output.joints[:, :24].detach().cpu().detach().numpy())
    return joints, vertices


def _getLabels(files: list[str]):
    """Returns a list of geometries that contain the file name as point cloud.

    Args:
        files (str): File name

    Returns:
        list(o3d.geometry.PointCloud()): List of all labels as point clouds
    """
    # Adjust if necessary
    # String manipulation to shorten the lables
    if files[0].endswith('gt'):
        redundant = files[0].replace("gt","")
        for i in range(1, len(files)):
            files[i] = files[i].replace(redundant, "")
    for i in range(len(files)):
        files[i]  = files[i].replace("DIPNet_", "")
        files[i]  = files[i].replace("05noise-", "")
        files[i]  = files[i].replace("AMASS_", "")
        files[i]  = files[i].replace("DIP_train", "")

    labels = []
    for file in files:
        labels.append(_text_3d(file, density=2, font_size=30))
    return labels


def _runVisualization(faces, joints, vertices, labels, capture=False, verbose=False, show_joints=False, show_labels=False, show_errors=False, sensors=[]):
    """Function that handles the visualization of the given vertices and labels.
    Creates a visualizer that runs through all frames for the n given sequences.

    Args:
        faces (numpy.ndarray[numpy.float64]): Float64 numpy array of shape (n, 3) that contains the model faces
        joints (list[joints]): List of joints for the respective SMPL poses
        vertices (list[vertices]): List of vertices for the respective SMPL poses
        labels (list[o3d.geoemtry.PointCloud]): List of pointclouds for each text label.
        capture (bool, optional): True if every frame should be captured as .png. Defaults to False.
        verbose (bool, optional): Increases verbosity if true. Defaults to False.
    """

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=2400, height=1200)

    if (capture):
        img_array = []

    if show_errors:
        #reds = cm.get_cmap('Reds', 256)
        greens = cm.get_cmap('Greens', 256)
        #blues = cm.get_cmap('Blues', 256)

    if show_joints:
        joint_meshes = [[o3d.geometry.TriangleMesh.create_sphere(radius=.025).paint_uniform_color(
            [0, 0.5, 0.5]) for _ in range(24)] for i in range(len(joints))]
    else:
        meshes = [o3d.geometry.TriangleMesh() for i in range(len(vertices))]

    if len(sensors) != 0:
        sensor_meshes = [[o3d.geometry.TriangleMesh.create_sphere(radius=.025).paint_uniform_color(
            [0.2, 0.2, 0.2]) for _ in range(len(sensors[j]))] for j in range(len(joints))]

    for frame in tqdm(range(len(vertices[0])), desc="Animating...", disable=not verbose):
        for i in range(len(vertices)):
            if show_joints:
                for j, mesh in enumerate(joint_meshes[i]):
                    mesh.translate(joints[i][frame][j] +
                                   [i * 2, 0, 0], relative=False)
                    if frame == 0:
                        vis.add_geometry(mesh)
                    else:
                        vis.update_geometry(mesh)
            else:
                meshes[i].vertices = o3d.utility.Vector3dVector(
                    vertices[i][frame])
                meshes[i].triangles = o3d.utility.Vector3iVector(faces)
                meshes[i].compute_vertex_normals()
                if show_errors:
                    meshes[i].paint_uniform_color([0.8, 0.8, 0.8])
                    if i != 0:
                        vertex_errors3d = np.absolute(np.subtract(vertices[0][frame], vertices[i][frame]))
                        vertex_errors = vertex_errors3d.mean(1)
                        # Normalize errors on deviation of 10 cm = 0.1; normalized = (x - min) / (max - min)
                        vertex_errors = vertex_errors / 0.1
                        # Manipulate vertex Colors
                        color = np.asarray(meshes[i].vertex_colors)
                        for idx, v in enumerate(color):
                            # Threshold to plot errors: 10 cm = 1.0, 2 cm = 0.2
                            if vertex_errors[idx] > 0.2:
                                #colors[idx] = reds(vertex_errors[idx])[:3]
                                color[idx] = greens(vertex_errors[idx])[:3]
                                #colors[idx] = blues(vertex_errors[idx])[:3]
                else:
                    meshes[i].paint_uniform_color(colors[i])
                meshes[i].translate([i * 2, 0, 0], relative=False)

                if len(sensors):
                    for k, sensor_mesh in enumerate(sensor_meshes[i]):
                        sensor_mesh.translate(meshes[i].vertices[sensors[i][k]], relative=False)
                        if frame == 0:
                            vis.add_geometry(sensor_mesh)
                        else:
                            vis.update_geometry(sensor_mesh)

                if frame == 0:
                    vis.add_geometry(meshes[i]) 
                else:
                    vis.update_geometry(meshes[i])
            if frame == 0 and show_labels:
                label_offset = labels[i].get_max_bound()[0] / 2
                labels[i].translate([i * 2 - label_offset, (1.1 if i % 2 else 1.4) , 0])
                vis.add_geometry(labels[i])

        vis.poll_events()
        vis.update_renderer()

        # save screen captures
        if (capture):
            vis.capture_screen_image(
                "capture/images/{}.png".format(frame), do_render=True)
            img = vis.capture_screen_float_buffer(do_render=True)
            img = (255.0 * np.asarray(img)).astype(np.uint8)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
    
    if (capture):
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        path = 'capture/' + datetime.now().strftime("%Y%m%d_%H%M") + '_screen_recording.avi'
        out = cv2.VideoWriter(path, fourcc, 60, size)

        for i in tqdm(range(len(img_array)), desc="Write Screen Recording...", disable=not verbose):
            out.write(img_array[i])
        out.release()
    vis.destroy_window()


if __name__ == "__main__":
    shape, files, is_rot_mat, capture, verbose, show_joints, show_labels, show_errors, vertex_key = _getArgs()

    if (capture and not os.path.exists("capture")):
        os.mkdir("capture", )
        os.mkdir("capture/images", )

    conf = Config(C.config_path)
    paths = conf["paths"]

    model = smplx.SMPL(model_path=paths["SMPL_male"])

    global_oris, poses = _loadPosesAndOris(files, is_rot_mat, verbose)

    faces = model.faces
    joints, vertices = _getJointsAndVertices(model, shape, global_oris, poses)
    labels = _getLabels(files)

    vertex_mask = []  
    for i in range(len(vertex_key)):
        vertex_list = conf["vertex_config"][vertex_key[i]]
        vertex_mask.append(torch.tensor(vertex_list))

    _runVisualization(faces, joints, vertices, labels,
                      capture, verbose, show_joints, show_labels, show_errors, vertex_mask)
