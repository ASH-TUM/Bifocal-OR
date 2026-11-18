import torch
import numpy as np
from typing import Any, Optional

from torchmetrics import Metric
from torchmetrics import Precision, Recall


def PrecisionOverClasses(num_classes: int = 5):
    return Precision(task='multiclass', num_classes=num_classes, average='none', ignore_index=-1)


def RecallOverClasse(num_classes: int = 5):
    return Recall(task='multiclass', num_classes=num_classes, average='none', ignore_index=-1)


class AccuracyStages(Metric):
    def __init__(self, num_stages=1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.num_stages = num_stages
        for s in range(self.num_stages):
            self.add_state(f"S{s + 1}_correct", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        target_flat = target.view(-1)
        valid_mask = target_flat != -1
        self.total += valid_mask.sum()
        for s in range(self.num_stages):
            pred_s = preds[s]
            while pred_s.dim() > 2:
                pred_s = pred_s.squeeze(0)
            pred_labels = torch.argmax(pred_s, dim=0).view(-1)
            s_correct = getattr(self, f"S{s + 1}_correct")
            s_correct += torch.sum((pred_labels == target_flat) & valid_mask)
            setattr(self, f"S{s + 1}_correct", s_correct)

    def compute(self):
        acc_list = []
        for s in range(self.num_stages):
            s_correct = getattr(self, f"S{s + 1}_correct")
            acc_list.append(s_correct.float() / self.total)
        return acc_list


def calc_average_over_metric(metric_list, normlist):
    for i in metric_list:
        metric_list[i] = np.asarray([0 if value == "None" else value for value in metric_list[i]])
        if normlist[i] == 0:
            metric_list[i] = 0
        else:
            metric_list[i] = metric_list[i].sum() / normlist[i]
    return metric_list


def create_print_output(print_dict, space_desc, space_item):
    msg = ""
    for key, value in print_dict.items():
        msg += f"{key:<{space_desc}}"
        for i in value:
            msg += f"{i:>{space_item}}"
        msg += "\n"
    msg = msg[:-1]
    return msg
