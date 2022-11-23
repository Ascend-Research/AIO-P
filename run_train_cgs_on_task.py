from constants import *
from params import *
import os
import copy
import torch
from model_src.predictor.gpi_family_data_manager import FamilyDataManager
from model_src.model_helpers import BookKeeper
import numpy as np
import pickle
from model_src.demo_functions import correlation_metrics


"""
Script for training/fine-tuning individual archs and making CGs from them.
Means we use OFAAdaptedCGHead, not TaskAdaptedCGModel or OFAAdaptedCGModel.
"""

def prepare_local_params(parser, ext_args=None):
    parser.add_argument("-family", required=False, type=str, default="ofa_pn",
                        help="architecture family")
    parser.add_argument("-task", required=False, type=str, default="hpe2d",
                        help="task type")
    parser.add_argument("-tag", required=False, type=str, default=None,
                        help="additional (recommended) tag for experiment folder")
    parser.add_argument("-start_idx", required=False, type=int,
                        default=0)
    parser.add_argument("-num_archs", required=False, type=int,
                        default=100)
    parser.add_argument("-cache_name", required=False, type=str,
                        default="hpe2d")
    parser.add_argument("-min", required=False, action="store_true",
                        default=False)
    parser.add_argument("-max", required=False, action="store_true",
                        default=False)
    parser.add_argument('-pt', required=False, type=int,
                        default=0)
    parser.add_argument('-chkpt', required=False, type=str, default=None,
                        help="checkpoint of weights to load")
    parser.add_argument('-skip', action="store_true", default=False,
                        help="whether to use skip-connections in the head")
    return parser.parse_known_args(ext_args)


def get_task_manager(task_name, cg_dict, task_params, book_keeper, skip=False, chkpt=None, cache_location=None):
    cg = cg_dict['compute graph']
    original_config = None
    if "original config" in cg_dict.keys():
        book_keeper.log("Original config detected")
        original_config = cg_dict['original config']

    if "hpe2d" in task_name:
        from tasks.pose_hg_3d.hg_3d_manager import HG3DManager
        task_manager = HG3DManager(name="Head", params=task_params, log_f=book_keeper.log)

        if skip or chkpt is not None:
            from model_src.multitask.skip_adapter import SkipDeconvHPEHead
            task_head = SkipDeconvHPEHead(task_name, hw=(task_manager.opts.output_h, task_manager.opts.output_w),
                                          joints=task_manager.opts.num_output)
        else:
            from model_src.multitask.task_adapter import DeconvHPEHead
            task_head = DeconvHPEHead(task_name, hw=(task_manager.opts.output_h, task_manager.opts.output_w),
                                          joints=task_manager.opts.num_output)
            
        input_dims = [task_manager.opts.input_h, task_manager.opts.input_w, 3]
        if original_config is None:
            from model_src.multitask.adapt_cg_framework import TaskAdaptedCGModel
            task_model = TaskAdaptedCGModel(base_cg=cg, task_head=task_head, input_dims=input_dims)
        elif skip or chkpt is not None:
            from model_src.multitask.adapt_ofa_head import OFAAdaptedCGHead
            task_model = OFAAdaptedCGHead(base_cg=cg, original_config=original_config, task_head=task_head,
                                          input_dims=input_dims, backprop=True, skip=skip)
        else:
            from model_src.multitask.adapt_ofa_framework import OFAAdaptedCGModel
            task_model = OFAAdaptedCGModel(base_cg=cg, original_config=original_config, task_head=task_head,
                                           input_dims=input_dims)

        task_manager.set_model(task_model)

        if chkpt is not None:
            book_keeper.load_model_checkpoint(task_head, skip_eval_perfs=True, checkpoint_file=chkpt)
    elif "detectron" in task_name:
        from tasks.detectron2.detectron2_manager import Detectron2Manager
        # Do not comment out these imports
        from tasks.detectron2.detectron2_adapter import Detectron2Adapter
        from tasks.detectron2.sem_seg_head_adapter import SemSegFPNHAdaptedHead

        task_manager = Detectron2Manager(task_params, exp_dir=cache_location, save=False, log_f=print)

        input_dims = [1024, 1024, 3]

        if chkpt is not None and 'resnet' in cg.name.lower():
            from model_src.multitask.fpn_adapter_base import FeaturePyramidAdapterBase
            from model_src.multitask.adapt_ofa_head import OFAAdaptedCGHead
            task_head = FeaturePyramidAdapterBase(name="Head", uniform_channels=True, hw=(256, 256), add_downsample=1)
            task_model = OFAAdaptedCGHead(base_cg=cg, original_config=original_config,
                                          task_head=task_head, input_dims=input_dims, swap_num=-1, backprop=2, skip=True)
        else:
            from model_src.multitask.fpn_adapter_detectron import FeaturePyramidDetectron
            from model_src.multitask.detectron2_ofa_head import Detectron2OFAHead
            task_head = FeaturePyramidDetectron(name="Head", hw=(256, 256), add_downsample=1)
            task_model = Detectron2OFAHead(base_cg=cg, original_config=original_config,
                                           task_head=task_head, input_dims=input_dims, freeze=2)
        task_manager.set_model(task_model, chkpt_file=chkpt)
    else:
        raise ValueError("Unknown task: {}".format(task_name))
    return task_manager


def main(params, unknown_params):

    name_list = [params.family, params.task]
    if params.tag is not None: name_list.append(params.tag)
    task_name = "_".join(name_list)
    task_tag = "_".join(name_list[1:])

    cache_location = P_SEP.join([CACHE_DIR, task_name])
    os.makedirs(cache_location, exist_ok=True)

    if params.min or params.max:
        if params.min and params.max:
            exp_name = "_".join([task_name, "min_max"])
        elif params.min:
            exp_name = "_".join([task_name, "min"])
        else:
            exp_name = "_".join([task_name, "max"])
    else:
        exp_name = "_".join([task_name, str(params.start_idx), str(params.start_idx + params.num_archs - 1)])

    book_keeper = BookKeeper(log_file_name=exp_name + ".txt",
                             model_name=exp_name,
                             saved_models_dir=params.saved_models_dir,
                             logs_dir=cache_location)
    book_keeper.log("Params: {}".format(params), verbose=False)

    data_manager = FamilyDataManager([params.family], log_f=book_keeper.log)
    cg_data = data_manager.load_cache_data(params.family)

    if params.min or params.max:
        cg_accs, min_idx, max_idx = [], [], []
        for entry in cg_data:
            cg_accs.append(entry['acc'])
        if params.min:
            min_idx.append(np.argmin(cg_accs))
        if params.max:
            max_idx.append(np.argmax(cg_accs))

        idx_to_consider = min_idx + max_idx

    else:
        cg_names = []
        for entry in cg_data:
            cg_names.append(entry["compute graph"].name)

        idx_to_consider = np.argsort(cg_names)[params.start_idx : params.start_idx + params.num_archs]

    cg_task_data, class_accs, task_scores = [], [], []
    for i, idx in enumerate(idx_to_consider):
        sel_cg_dict = cg_data[idx]
        class_accs.append(sel_cg_dict['acc'])
        book_keeper.log("Network: {}".format(sel_cg_dict['compute graph'].name))
        book_keeper.log("Classification Acc: {}".format(class_accs[-1]))

        task_manager = get_task_manager(task_tag, sel_cg_dict, unknown_params, book_keeper, skip=params.skip, chkpt=params.chkpt, cache_location=cache_location)

        metric_dict = task_manager.train(eval_test=True)
        book_keeper.log(metric_dict) 

        test_metric = task_manager.test_metric

        new_cg_dict = copy.deepcopy(sel_cg_dict)

        if "hpe" in params.task:
            task_scores.append(metric_dict[test_metric])
            new_cg_dict[test_metric] = task_scores[-1]
            if isinstance(task_manager.model, torch.nn.DataParallel):
                task_manager.model.module.build_overall_cg()
            else:
                task_manager.model.build_overall_cg()
        else:
            obj_det_AP = metric_dict.get('bbox',{}).get('AP')
            sem_seg_mIoU = metric_dict.get('sem_seg',{}).get('mIoU')
            inst_seg_AP = metric_dict.get('segm',{}).get('AP')
            pan_seg_PQ = metric_dict.get('panoptic_seg',{}).get('PQ')

            if obj_det_AP is not None: new_cg_dict['obj_det_AP'] = obj_det_AP
            if sem_seg_mIoU is not None: new_cg_dict['sem_seg_mIoU'] = sem_seg_mIoU
            if inst_seg_AP is not None: new_cg_dict['inst_seg_AP'] = inst_seg_AP
            if pan_seg_PQ is not None: new_cg_dict['pan_seg_PQ'] = pan_seg_PQ
            new_cg_dict['detectron2_metrics'] = metric_dict

            task_scores.append(pan_seg_PQ or inst_seg_AP or obj_det_AP) 
            
        if isinstance(task_manager.model, torch.nn.DataParallel):
            new_cg_dict['compute graph'] = task_manager.model.module.overall_cg
        else:
            if "hpe" in params.task:
                new_cg_dict['compute graph'] = task_manager.model.overall_cg
            else:
                new_cg_dict = {**new_cg_dict, **task_manager.build_cgs(return_dict=True)}

        cg_task_data.append(new_cg_dict)

        if params.min or params.max:
            cache_file_name = P_SEP.join([cache_location, exp_name + ".pkl"])
            with open(cache_file_name, "wb") as f:
                pickle.dump(cg_task_data, f, protocol=4)
        else:
            suffix = "_".join([task_name, str(params.start_idx)])
            new_suffix = "_".join([suffix, str(params.start_idx + i)])
            cache_file_name = P_SEP.join([cache_location, new_suffix + ".pkl"])
            with open(cache_file_name, "wb") as f:
                pickle.dump(cg_task_data, f, protocol=4)
            
            if i > 0:
                old_suffix = "_".join([suffix, str(params.start_idx + i - 1)])
                old_cache_file_name = P_SEP.join([cache_location, old_suffix + ".pkl"])
                os.remove(old_cache_file_name)

    if len(class_accs) > 1:
        book_keeper.log("Spearman Correlation between "
                        "classification accuracy and task {}: {}".format(test_metric,
                                                                         correlation_metrics(class_accs, task_scores,
                                                                                         printfunc=book_keeper.log)[0]))


if __name__ == "__main__":
    _parser = prepare_global_params()
    params, unknown_params = prepare_local_params(_parser)
    main(params, unknown_params)
