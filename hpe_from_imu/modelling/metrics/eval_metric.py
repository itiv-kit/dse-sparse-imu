import torch
from hpe_from_imu.dataloader import RemoveAcc, ToFullSMPL
from hpe_from_imu.evaluation import PoseEvaluator
from poutyne import Metric
from tqdm import tqdm


class MeshMetric(Metric):
    def __init__(self):
        super().__init__()
        self.__name__ = [
            'SIP Error (deg) mean', 'SIP Error (deg) std',
            'Angular Error (deg) mean', 'Angular Error (deg)std',
            'Positional Error (cm) mean', 'Positional Error (cm) std',
            'Mesh Error (cm) mean', 'Mesh Error (cm) std',
            'Jitter Error (100m per s3) mean', 'Jitter Error (100m|s3) std',
        ]
        self._outward_transforms = [RemoveAcc(), ToFullSMPL()]
        self.reset()

    def update(self, y_pred, y_true):
        self._update(y_pred, y_true)

    def _update(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> None:
        self.y_pred_list.append(y_pred.detach().cpu().numpy())
        self.y_true_list.append(y_true.detach().cpu().numpy())

    def compute(self):
        errs = []
        evaluator = PoseEvaluator()
        for y_, y in tqdm(list(zip(self.y_pred_list, self.y_true_list))):
            y__full = self._apply_outwards_transforms(torch.from_numpy(y_))
            y_full = self._apply_outwards_transforms(torch.from_numpy(y))
            errs.append(evaluator.eval(y__full, y_full))
        return torch.stack(errs).mean(dim=0).flatten()

    def reset(self):
        self.y_true_list = []
        self.y_pred_list = []

    def _apply_outwards_transforms(self, y):
        if self._outward_transforms is not None:
            for transformation in self._outward_transforms:
                y = transformation(y)
        return y
