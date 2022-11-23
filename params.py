import os
import argparse
from os.path import sep as P_SEP


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = P_SEP.join([BASE_DIR, "cache"])
DATA_DIR = P_SEP.join([BASE_DIR, "data"])
LIBS_DIR = P_SEP.join([BASE_DIR, "libs"])
SAVED_MODELS_DIR = P_SEP.join([BASE_DIR, "saved_models"])
LOGS_DIR = P_SEP.join([BASE_DIR, "logs"])


if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
if not os.path.exists(LIBS_DIR): os.makedirs(LIBS_DIR)
if not os.path.exists(SAVED_MODELS_DIR): os.makedirs(SAVED_MODELS_DIR)
if not os.path.exists(LOGS_DIR): os.makedirs(LOGS_DIR)


def prepare_global_params():

    parser = argparse.ArgumentParser()

    parser.add_argument("-device_str", type=str, required=False,
                        default=None)
    parser.add_argument("-seed", required=False, type=int,
                        default=12345)
    parser.add_argument("-max_gradient_norm", required=False, type=float,
                        default=5.0)
    parser.add_argument("-logs_dir", required=False, type=str,
                        default=LOGS_DIR)
    parser.add_argument("-saved_models_dir", required=False, type=str,
                        default=SAVED_MODELS_DIR)
    parser.add_argument("-saved_model_file", required=False, type=str,
                        default=P_SEP.join([SAVED_MODELS_DIR, "default_model.pt"]))
    parser.add_argument("-allow_parallel", required=False, action="store_true",
                        default=False)

    return parser
