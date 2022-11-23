import pickle
from params import *


def load_gpi_ofa_pn_mbv3_src_data(cache_file=P_SEP.join([CACHE_DIR, "gpi_ofa_src_data.pkl"])):
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    return data


def load_gpi_nb101_src_data(cache_file=P_SEP.join([CACHE_DIR, "gpi_nb101_src_data.pkl"])):
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    return data


def load_gpi_ofa_resnet_src_data(cache_file=P_SEP.join([CACHE_DIR, "gpi_ofa_resnet_src_data.pkl"])):
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    return data

