from constants import *
from params import *
import os
import numpy as np
import pickle
import re
from tqdm import tqdm

"""
Script for profiling architectures for FLOPS (floating point operations per second) and params.
Supported for HPE and Detectron2 architectures.
IMPORTANT: cache_file must be directly from run_train_cgs_on_task (eg. ends in 1000_1999.pkl). 
this should be be a gpi_ofa_*_comp_graph_cache.pkl file!
"""

def get_final_c(cg, original_config):
    name = cg.name.lower()
    if "mbv3" in name:
        return 1152
    elif "resnet" in name:
        original_config_key = original_config[1]["w"][-1]
        return [1328, 1640, 2048][original_config_key]
    else:
        return 1664


def prepare_local_params(parser, ext_args=None):
    parser.add_argument(
        "-profiler",
        required=False,
        nargs="+",
        help="List of metrics to profile",
        default=["flops", "params"],
    )
    parser.add_argument(
        "-cache_file",
        required=False,
        type=str,
        default=None,
        help="Path to cache file with compute graphs",
    )
    parser.add_argument(
        "-chkpt_file",
        required=False,
        type=str,
        default=None,
        help="Detectron model checkpoint file",
    )
    parser.add_argument(
        "-task", required=False, type=str, default="detectron", help="task type"
    )
    parser.add_argument(
        "-skip",
        action="store_true",
        default=False,
        help="whether to use skip-connections in the head",
    )
    parser.add_argument(
        "-reprofile",
        action="store_true",
        default=False,
        help="allow reprofiling an already profiled cache",
    )

    return parser.parse_known_args(ext_args)


def load_cache(cache_file):
    assert cache_file.endswith(".pkl"), "Must be a .pkl file"
    # Load in the cache from path
    print("Loading", cache_file)
    with open(cache_file, "rb") as f:
        data = pickle.load(f)
    return data, cache_file


def save_to_json(data, output_file):
    def convert(o):
        return str(o)

    with open(output_file, "w+") as outfile:
        import json

        json.dump(data, outfile, indent=4, default=convert)
    print("Saved json to", output_file)


def set_cg_model(params, task_manager, cg_dict):
    """
    Given a compute graph dict, instantiate the model
    """
    cg = cg_dict["compute graph"]
    original_config = None
    if "original config" in cg_dict.keys():
        print("Original config detected")
        original_config = cg_dict["original config"]

    if "hpe2d" in params.task:
        actual_dims = (8, 8, get_final_c(cg, original_config))

        if params.skip:
            from model_src.multitask.skip_adapter import SkipDeconvHPEHead

            task_head = SkipDeconvHPEHead(
                params.task,
                hw=(task_manager.opts.output_h, task_manager.opts.output_w),
                joints=task_manager.opts.num_output,
            )
        else:
            from model_src.multitask.task_adapter import DeconvHPEHead

            task_head = DeconvHPEHead(
                params.task,
                hw=(task_manager.opts.output_h, task_manager.opts.output_w),
                joints=task_manager.opts.num_output,
            )

        input_dims = [task_manager.opts.input_h, task_manager.opts.input_w, 3]
        if original_config is None:
            from model_src.multitask.adapt_cg_framework import TaskAdaptedCGModel

            task_model = TaskAdaptedCGModel(
                base_cg=cg, task_head=task_head, input_dims=input_dims
            )
        elif params.skip:
            from model_src.multitask.adapt_ofa_head import OFAAdaptedCGHead

            task_model = OFAAdaptedCGHead(
                base_cg=cg,
                original_config=original_config,
                task_head=task_head,
                input_dims=input_dims,
                backprop=True,
                skip=params.skip,
            )
        else:
            from model_src.multitask.adapt_ofa_framework import OFAAdaptedCGModel

            task_model = OFAAdaptedCGModel(
                base_cg=cg,
                original_config=original_config,
                task_head=task_head,
                input_dims=input_dims,
            )
            task_model.task_head.build(resolution=actual_dims)

        task_manager.set_model(task_model)

    elif "detectron" in params.task:
        from model_src.multitask.fpn_adapter_detectron import (
            FeaturePyramidDetectron,
        )
        from model_src.multitask.detectron2_ofa_head import Detectron2OFAHead

        input_dims = [1024, 1024, 3]

        task_head = FeaturePyramidDetectron(
            name="Head", hw=(256, 256), add_downsample=1
        )
        task_model = Detectron2OFAHead(
            base_cg=cg,
            original_config=original_config,
            task_head=task_head,
            input_dims=input_dims,
            freeze=2,
        )

        if params.chkpt_file is None: 
            print("NO CHECKPOINT FILE PROVIDED")
        else:
            print("Using checkpoint file:", params.chkpt_file)
        task_manager.set_model(task_model, chkpt_file=params.chkpt_file, chkpt_strict=False)


def main(params, unknown_params):
    print(f">>> Profiling params:", params)
    profile(params, unknown_params)


def get_task_manager(task, task_params, cache_location):
    if "hpe2d" in task:
        from tasks.pose_hg_3d.hg_3d_manager import HG3DManager

        task_manager = HG3DManager(name="Head", params=task_params)
    elif "detectron" in task:
        from tasks.detectron2.detectron2_manager import Detectron2Manager

        # Do not comment out these imports
        from tasks.detectron2.detectron2_adapter import Detectron2Adapter
        from tasks.detectron2.sem_seg_head_adapter import SemSegFPNHAdaptedHead

        task_manager = Detectron2Manager(
            unknown_params, exp_dir=cache_location, save=False
        )
    else:
        raise Exception("Not a valid task")

    return task_manager


def profile(params, unknown_params):
    assert params.cache_file is not None, "Cache file path must not be None"
    cache_file = params.cache_file

    if os.path.isdir(cache_file):
        print("Folder detected, reading in files")
        cache_contents = os.listdir(cache_file)
        filtered_files = [
            os.path.join(cache_file, file)
            for file in cache_contents
            if file.endswith(".pkl")
        ]
        for file in filtered_files:
            params.cache_file = file
            profile(params, unknown_params)
        return  # exit

    cg_data, cache_file = load_cache(params.cache_file)
    cache_location = os.path.join(os.path.dirname(cache_file), "profiler")
    os.makedirs(cache_location, exist_ok=True)

    new_cg_data, class_accs = [], []

    for sel_cg_dict in tqdm(cg_data, desc="Profiling", total=len(cg_data)):
        task_manager = get_task_manager(params.task, unknown_params, cache_location)

        class_accs.append(sel_cg_dict["acc"])
        print("Network: {}".format(sel_cg_dict["compute graph"].name))
        print("Classification Acc: {}".format(class_accs[-1]))

        profile_dict = {}

        # SKIP already evaluated metrics
        profiler_metrics = []
        if not params.reprofile:
            for metric in params.profiler:
                if any(key.startswith(metric) for key in sel_cg_dict.keys()):
                    print(f"[SKIPPING] {metric} has already been profiled")
                else:
                    profiler_metrics.append(metric)
        else:
            profiler_metrics = params.profiler
        
        print("Profiler metrics:", profiler_metrics)
        profile_dict = {}
        if len(profiler_metrics) > 0:
            # Check that this cg has not already been profiled (ie. has a flops_ key)
            set_cg_model(params, task_manager, sel_cg_dict)
            profile_dict = task_manager.profiler(profiler_metrics)
            print("profile_dict", profile_dict)
        else:
            print("[SKIPPING] No profiler metrics founds! Skipping this CG")

        # Add profile key-value pairs to the dict
        new_cg_dict = {**sel_cg_dict, **profile_dict}
        new_cg_data.append(new_cg_dict)

    # save_to_json(new_cg_data, cache_file + ".json")  # For Debugging
    print("OVERWRITING CACHE FILE:", cache_file)
    with open(cache_file, "wb") as f:
        pickle.dump(new_cg_data, f, protocol=4)


if __name__ == "__main__":
    _parser = prepare_global_params()
    params, unknown_params = prepare_local_params(_parser)
    main(params, unknown_params)
