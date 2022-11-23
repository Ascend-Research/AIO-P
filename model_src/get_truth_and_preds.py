import torch
from constants import *
import numpy as np


def get_reg_truth_and_preds(model, loader, fwd_func, printfunc=print):

    labels = []
    preds = []
    flops = []
    with torch.no_grad():
        for batch in loader:
            batch_labs = batch[DK_BATCH_TARGET_TSR]
            labels += batch_labs.detach().tolist()
            batch_preds = fwd_func(model, batch)
            preds += batch_preds.detach().tolist()
            flops += batch[DK_BATCH_FLOPS].cpu().tolist()

    labels = np.array(labels).squeeze()
    preds = np.array(preds).squeeze()
    flops = np.array(flops).squeeze()

    return labels, preds, flops


class RescaleLoss(torch.nn.Module):
    def __init__(self, base_loss, model, weight=0.):
        super(RescaleLoss, self).__init__()
        self.base_loss = base_loss
        self.model = model
        self.weight = weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        sparse_penalty = self.weight * torch.sum(torch.abs(self.model.b))
        return self.base_loss(input=input, target=target) + sparse_penalty
