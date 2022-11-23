import torch
import pickle
from params import *
from constants import *
from utils.misc_utils import one_hot_list_to_idx


def load_nb101_acc_lat_dict(cache_file=P_SEP.join([CACHE_DIR, "nb101_acc_lat_dict.pkl"]),
                            log_f=print):
    log_f("Loading NB101 acc lat dict")
    with open(cache_file, "rb") as f:
        perf_dict = pickle.load(f)
    log_f("{} block performance values loaded".format(len(perf_dict)))
    return perf_dict


def load_nb101_processed_data(cache_file=P_SEP.join([CACHE_DIR, "nb101_processed_data.pkl"]),
                              log_f=print, convert_to_compact_indices=True,
                              convert_to_tsr=True):
    log_f("Reading NB101 processed data")
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    data_list = data[DATA]
    unique_node2feature = data[UNIQUE_NODE2FEATURE]
    unique_feature_list = data[UNIQUE_FEATURE_LIST]
    # Convert 1-hot to compact indices
    if convert_to_compact_indices:
        unique_feature_list = [(t[0], one_hot_list_to_idx(t[1])) for t in unique_feature_list]
        unique_node2feature = {k: one_hot_list_to_idx(v) for k, v in unique_node2feature.items()}
    if convert_to_tsr:
        unique_feature_list = [(t[0], torch.LongTensor(t[1] if isinstance(t[1], list) else [t[1]]))
                                for t in unique_feature_list]
        unique_node2feature = {k: torch.LongTensor(v if isinstance(v, list) else [v])
                                for k, v in unique_node2feature.items()}
    return data_list, unique_feature_list, unique_node2feature


def load_nb101_train_test_data(cache_file=P_SEP.join([CACHE_DIR, "nb101_train_test_data.pkl"])):
    # Loads pre-processed train/test data splits
    with open(cache_file, "rb") as f:
        _data = pickle.load(f)
    train_dev_data, test_data = _data[DATA]
    return train_dev_data, test_data
