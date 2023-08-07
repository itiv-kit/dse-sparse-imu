# %%
import torch

from evaluation import *

# %%
dip_y = torch.load("dip_gt_10_5-1")
dip_y_ = torch.load("dip_pred_10_5-1")

# tc_y = torch.load("tc_gt_10_5-1")
# tc_y_ = torch.load("tc_pred_10_5-1")
# s_10-02 from 11267 to 14874

# %%


def batchify(data, batchsize):
    nbatch = data.size(0) // batchsize
    data = data.narrow(0, 0, nbatch * batchsize)
    data = data.view(nbatch, -1, 24, 3)
    return data


def eval_in_batches(gt, pred, batchsize):
    errs = []
    evaluator = PoseEvaluator()
    for batch_gt, batch_pred in zip(batchify(gt, batchsize), batchify(pred, batchsize)):
        errs.append(evaluator.eval(batch_pred, batch_gt))
    return torch.stack(errs).view(5, 2, -1).mean(2)


# %%
dip_errors = eval_in_batches(dip_y, dip_y_, 2000)
# tc_errors = eval_in_batches(tc_y, tc_y_, 2000)
print(dip_errors)
# print(tc_errors)
