# Feature: Customized Vertex and Joint Configuration

This file is a documentation to synthesize a custom vertex and joint configuration from DIP IMU dataset. Other datasets are not implemented at this point.

# Background
Deep Inertial Poser (DIP) used the joints and vertices:
```
ji_mask = torch.tensor([18, 19, 4, 5, 15, 0]) 
vi_mask = torch.tensor([1961, 5424, 1176, 4662, 411, 3021])
```
while the order describes the sensor positions and is namely:
```
[left_lower_wrist, right_lower_wrist, left_lower_leg, right_loewr_leg, head, back]
```
This is reflected in both data['ori'] and data['acc'] accordingly.
For customized vertex and joint configurations it is recommended to close the list with the pelvis joint and vertex.
The pelvis is used as root joint and all other joints are normalized on it.
Further each vertex should match a joint in the order of the list. 

## SMPL Parameter
According to  [Meshcapade's Wiki](https://meshcapade.wiki/SMPL#skeleton-layout) the skeletral configuration is indexed as follows:
```
# 0: 'Pelvis',     3: 'Spine1',       6: 'Spine2',    9: 'Spine3',    12: 'Neck',     15: 'Head',
# 1: 'L_Hip',      4: 'L_Knee',       7: 'L_Ankle',  10: 'L_Foot',
# 2: 'R_Hip',      5: 'R_Knee',       8: 'R_Ankle',  11: 'R_Foot',
# 13: 'L_Collar',  16: 'L_Shoulder',  18: 'L_Elbow',  20: 'L_Wrist',
# 14: 'R_Collar',  17: 'R_Shoulder',  19: 'R_Elbow',  21: 'R_Wrist',
# 22: 'L_Hand',
# 23: 'R_Hand'
```
For the vertex indices it is important to be aware that we use SMPL and NOT SMPL-X.
The SMPL and SMPL-H (MANO) uses the same Mesh but is not compatible to SMPL-X, even though we use the 'smplx'-package. 
```
from smplx import SMPL
```
A SMPL vertex segmentation can be found [here](https://meshcapade.wiki/assets/SMPL_body_segmentation/smpl/smpl_vert_segmentation.json).
# Setup
## Prerequirements
You followed the Setup of [README](README.md) and your dataset directory looks at least like this:
```
└── dataset_raw/
    ├── DIP_IMU_and_Others
    │   ├── DIP_IMU
    │   │   ├── s_01
    │   │   ├── ...
    │   │   └── s_10
    │   └── DIP_IMU_nn
```

## Configuration

First, it is necessary to define the new location of storage for the synthesized data. Open [configuration/configuration.py](configuration/configuration.py) and find the class "_Path()". Enter a custom path:
``` 
PATH_KEY = join_path(data_dir_raw, "FOLDER_NAME")
PATH_KEY_preprocessed = join_path(data_dir_work, "FOLDER_NAME")
```
You can use the entry "Custom_1_path" as an example. It is recommended to call the second variable similar to the `PATH_KEY` with the ending `preprocessing`. However, the names can be choosen freely.
In the class "_DATASET_PATHS()" give your custom dataset a name and enter the path from "PATH_KEY_preprocessed"
Continue at "dataset_base" and enter the custom dataset name with the dataset origin, like in the examples.
Repeat these steps for each dataset you use (AMASS, DIP_train, and DIP_test).

Further find the class "_VERTEX_CONFIG()". Enter the customized vertex configuration as a list:
```
VERTEX_KEY = [vertex_index, ...] 
```
"Custom_1_vertex" can be used as example.

To adjust a customized joint configuration, note your setting under "_JOINT_CONFIG()"
```
JOINT_KEY = [joint_index, ...] 
```
"Custom_1_joint" can be used as example.

Save and run [configuration/configuration.py](configuration/configuration.py):
```
python configuration/configuration.py
```
You can now find your new configuration entries in [configuration/config.yaml](configuration/config.yaml).

# Creation of custom synthesized dataset and preprocessing

After the configuration has been done, data can be synthesized. Open and customize the main of[preprocess.py](preprocess.py). The funtion `custom_synthesize_DIP` takes the previously generated keys as input parameter and creates a dataset with the customized vertices and joints at the `PATH_KEY`-location.
```python
if __name__ == '__main__':
    custom_synthesize_AMASS(PATH_KEY, VERTEX_KEY, JOINT_KEY)
    preprocess_AMASS_TP_custom_synth(PATH_KEY, PATH_KEY_preprocessed)
    custom_synthesize_DIP(PATH_KEY, VERTEX_KEY, JOINT_KEY)
    preprocess_DIP_TP_custom_synth(PATH_KEY, PATH_KEY_preprocessed)
```
Now save and run [preprocess.py](preprocess.py):
```
python preprocess.py
```
Output (shortened):
```
Synthesized $PATH_KEY dataset saved at $/home/.../data/dataset_raw/DIP_IMU_FOLDER_NAME
```
The script saved the synthetic data `data/dataset_raw/DIP_IMU_FOLDER_NAME`.


# Use custom synthesized dataset for training and evaluation
If the dataset is created, a customized model can be created.
You can use a model configration file (.yml) from the forlder [configuration/experiments](configuration/experiments) and modify the dataset section to your need. Enter the dataset name from the config.yml file as train, finetune or testset.

Now run the the training file from a terminal.
```
python train.py -c CUSTOM_NET.yml
```

# Generate a Visual Example
Note: currently only DIP-IMU examples can be generated.

Find the `generate-example.py` script in the main directory and specify your example in the main.
The variables `sub, mot, spec` stores subject, motion, and specifier e.g. like `10, 1, '_a'` or `10, 2, ''`.
In case you are using the default node configuration comment out the line
```
# custom_example("DIP_REF", sub, mot, spec)
```
and run the script like explained in the [README](README.md) 
If you want to capture the animation use `-c` flag.
To display the lables use the `-l` flag. 
In case you expect different lables check [visualize_sequence.py](visualize_sequence.py#L206).