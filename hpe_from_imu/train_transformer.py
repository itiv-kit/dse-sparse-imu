# %%
import poutyne
import torch
from torchsummary import summary

from hpe_from_imu.evaluation import *
from hpe_from_imu.modelling import *
from hpe_from_imu.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_past_frames = 10
num_future_frames = 5
epochs = 1
total_frames = num_future_frames + num_past_frames + 1
suffix = f"{num_past_frames}_{num_future_frames}-{epochs}"

# %%
dip_train_data = torch.load("/data/uxukh/dataset_work/DIP_IMU/train.pt")
dip_test_data = torch.load("/data/uxukh/dataset_work/DIP_IMU/test.pt")
# tc_test_data = torch.load("/data/uxukh/dataset_work/TotalCapture/test.pt")


def prep_data(data):
    acc = data["acc"]
    ori = data["ori"]
    pose = data["pose"]

    x, y = [], []

    for i in range(len(acc)):
        imu = torch.cat((acc[i], ori[i].view(-1, 6, 9)), 2)
        imu_split = unfold_to_sliding_windows(
            imu.view(1, -1, 72), num_past_frames, num_future_frames).reshape(-1, total_frames, 6, 12)
        x.extend(imu_split)
        y.extend(pose[i].view(-1, 24, 3))

    return torch.stack(x).to(device), torch.stack(y).to(device)


dip_train_x, dip_train_y = prep_data(dip_train_data)
dip_test_x, dip_test_y = prep_data(dip_test_data)
# tc_test_x, tc_test_y = prep_data(tc_test_data)

# pose_dip = dip_test_data["pose"]
# lens = [len(elem) for elem in pose_dip]


# def get_bounds(array, elem):
#     start = 0
#     for i in range(elem):
#         start += array[i]
#     return start, start + array[elem]


# print(lens)
# print(get_bounds(lens, 4))

# %%
has_trained = False

net = PoseWrapper(num_frames=total_frames)
summary(net, (total_frames, 6, 12), verbose=2, col_names=("input_size",
        "output_size", "num_params", "kernel_size", "mult_adds"))
model = poutyne.Model(net, optimizer="Adam",
                      loss_function="mse", device=device)
if not has_trained:
    model.fit(dip_train_x, dip_train_y, epochs=epochs)
    model.save_weights(f"transformer_{suffix}.weights")
else:
    model.load_weights(f"transformer_{suffix}.weights")

# %%
dip_test_y_ = model.predict(dip_test_x, convert_to_numpy=False)
# tc_test_y_ = model.predict(tc_test_x, convert_to_numpy=False)

# %%
dip_y = dip_test_y.cpu()
dip_y_pred = dip_test_y_.cpu()
torch.save(dip_y, f"dip_gt_{suffix}")
torch.save(dip_y_pred, f"dip_pred_{suffix}")

# tc_y = tc_test_y.cpu()
# tc_y_pred = tc_test_y_.cpu()
# torch.save(tc_y, f"tc_gt_{suffix}")
# torch.save(tc_y_pred, f"tc_pred_{suffix}")

# %%
