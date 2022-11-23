from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import torch
import torch.utils.data
from tasks.pose_hg_3d.lsp_dataloader import LSPDataset
from tasks.pose_hg_3d.lib.opts import opts  
from tasks.pose_hg_3d.lib.model import get_optimizer

from tasks.pose_hg_3d.lib.datasets.mpii import MPII
from tasks.pose_hg_3d.lib.datasets.lsp_extended import LSPExtended
from tasks.pose_hg_3d.lib.train import train, val
import copy

from tasks.task_manager_base import BaseTaskManager
from utils.model_utils import measure_gpu_latency

dataset_factory = {
    "mpii": MPII,
    "lsp_extended": LSPExtended,
    "lsp_cache": LSPDataset,
}

task_factory = {
    '2d': (train, val),
}


class HG3DManager(BaseTaskManager):
    def __init__(self, task='2d', name="default", params=None, log_f=print):
        super(HG3DManager, self).__init__(log_f=log_f)

        opt = opts()
        self.opts = opt.parse(params)
        opts.exp_id = name

        # set random seed
        torch.manual_seed(self.opts.random_seed)
        random.seed(self.opts.random_seed)
        np.random.seed(self.opts.random_seed)

        self.opts.device = torch.device('cuda:{}'.format(self.opts.gpus[0]))

        self.dataset = dataset_factory[self.opts.dataset]
        self.train_f, self.val_f = task_factory[task]
        self.train_metric = "train_PCK"
        self.train_dict[self.train_metric] = -1
        self.test_metric = "val_PCK" 
        self.test_dict[self.test_metric] = -1
        self.test_preds = None

        self.train_loader = torch.utils.data.DataLoader(
            self.dataset(self.opts, 'train'),
            batch_size=self.opts.batch_size * len(self.opts.gpus),
            shuffle=True, 
            num_workers=self.opts.num_workers,
            pin_memory=True,
        )

        self.val_loader = torch.utils.data.DataLoader(
            self.dataset(self.opts, 'val'),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )

        self.optimizer = None
        self.scheduler = None

    def set_model(self, model):


        self.optimizer = get_optimizer(model, self.opts.optim_type, self.opts.lr, self.opts.weight_decay,
                                       self.opts.momentum)
        if self.opts.lr_cosine:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.opts.num_epochs,
                                                                        eta_min=0.00001)

        if len(self.opts.gpus) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=self.opts.gpus).cuda(self.opts.device)
        else:
            self.model = model.cuda(self.opts.device)

        self.model.optim = self.optimizer
        self.model.set_optim_params()

    def train(self, eval_test=False):

        self.log_f("Training for {} epochs".format(self.opts.num_epochs))
        for epoch in range(self.opts.num_epochs):
            log_dict_train, _ = self.train_f(epoch, self.opts, self.train_loader, self.model, self.optimizer)
            for k, v in log_dict_train.items():
                self.log_f('{} {:8f} | '.format(k, v))
            if self.train_dict[self.train_metric] < log_dict_train[self.train_metric]:
                self.train_dict = log_dict_train

            self.log_f('\n')
            if self.scheduler is not None:
                self.scheduler.step()
            elif epoch in self.opts.lr_step:
                lr = self.opts.lr * (0.1 ** (self.opts.lr_step.index(epoch) + 1))
                self.log_f('Drop LR to', lr)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

        if eval_test:
            self.eval()
        return {**self.train_dict, **self.test_dict}

    def eval(self):
        log_dict_test, preds = self.val_f(0, self.opts, self.val_loader, self.model)
        self.test_preds = preds
        msg = "Evaluation: "
        for key in log_dict_test.keys():
            msg += "{}: {} | ".format(key, log_dict_test[key])

        self.log_f(msg)
        if self.test_dict[self.test_metric] < log_dict_test[self.test_metric]:
            self.test_dict = log_dict_test
            self.best_model = copy.deepcopy(self.model)
        return self.test_dict

    def profiler(self, profiler_metrics,  log_f=print):
        """
        Profile for FLOPs and Params
        """
        from utils.model_utils import device

        profiler_output = {}
        rand_input = torch.randn(1, 3, 256, 256).to(device()) # BxCxWxH

        if "flops" in profiler_metrics or "params" in profiler_metrics:
            from thop import profile
            macs, params = profile(self.model, inputs=(rand_input,))

            if "flops" in profiler_metrics:
                flops = macs * 2
                flops /= 1e9
                profiler_output["flops_hpe"] = flops
            
            if "params" in profiler_metrics:
                params /= 1e6
                profiler_output["params_hpe"] = params

        return profiler_output
