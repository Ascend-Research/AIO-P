import math
import torch
import random
import numpy as np
import torch.nn as nn


DEVICE_STR_OVERRIDE = None


def device(device_id="cuda", ref_tensor=None):
    if ref_tensor is not None:
        return ref_tensor.get_device()
    if DEVICE_STR_OVERRIDE is None:
        return torch.device(device_id if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(DEVICE_STR_OVERRIDE)


def measure_gpu_latency(inf_func, m_ignore_runs=10, n_reps=100):
    from utils.math_utils import mean, variance
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []
    # GPU-WARM-UP
    for _ in range(m_ignore_runs):
        inf_func()
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(n_reps):
            starter.record()
            inf_func()
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) # in milliseconds
            timings.append(curr_time)
    return mean(timings), variance(timings)


def set_random_seed(seed, log_f=print):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    log_f('My seed is {}'.format(torch.initial_seed()))
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        log_f('My cuda seed is {}'.format(torch.cuda.initial_seed()))



def model_save(file_path, data):
    torch.save(data, file_path)


def model_load(file_path):
    return torch.load(file_path, map_location=lambda storage,loc:storage)


def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.},
            {'params': decay, 'weight_decay': weight_decay}]


def torch_geo_batch_to_data_list(batch_tsr, last_node_idx_list,
                                 batch_indicator=None):
    """
    :param batch_tsr: PyTorch geo batch object
    :param last_node_idx_list: global batch indices of the last nodes of each graph
    :param batch_indicator: only for correctness checking
    :return: list of graphs recovered from the batch
    """
    rv = []
    batch_idx = 0
    graph_idx = 0
    if batch_indicator is not None:
        batch_indicator = batch_indicator.tolist()
    while batch_idx < batch_tsr.shape[0]:
        inst_delim = last_node_idx_list[graph_idx] + 1
        assert batch_tsr.shape[0] >= inst_delim, \
            "Specified inst end: {}, but entire batch is only of size: {}".format(inst_delim, batch_tsr.shape[0])
        if batch_indicator is not None:
            assert all(v == batch_indicator[batch_idx] for v in batch_indicator[batch_idx: inst_delim]), \
                "Batch indicator check failure, indicator list: {}\n Start, end: {}, {}".\
                    format(batch_indicator, batch_idx, inst_delim)
        rv.append(batch_tsr[batch_idx: inst_delim, :])
        batch_idx = inst_delim  # start of next batch
        graph_idx += 1
    return rv


def get_activ_by_name(_name):
    if _name is None or _name.lower() == "none":
        return None
    elif _name.lower() == "relu":
        return nn.ReLU()
    elif _name.lower() == "relu6":
        return nn.ReLU6()
    elif _name.lower() == "tanh":
        return nn.Tanh()
    elif _name.lower() == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError("Unknown activ name: {}".format(_name))

