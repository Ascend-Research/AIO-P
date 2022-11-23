from constants import *
from params import *
import os
from model_src.predictor.gpi_family_data_manager import FamilyDataManager
from model_src.model_helpers import BookKeeper
import numpy as np

"""
Script for training the shared head only. What does that mean?
Means we use OFAAdaptedCGHead, not TaskAdaptedCGModel or OFAAdaptedCGModel.
"""

class cast_backprop(argparse.Action):
    def __call__(self, parser, args, value, option_string=None):
        if value in [0,1]:
            value = bool(value)
        setattr(args, self.dest, value)

def prepare_local_params(parser, ext_args=None):
    parser.add_argument("-family", required=False, type=str, default="ofa_pn",
                        help="architecture family")
    parser.add_argument("-task", required=False, type=str, default="hpe2d",
                        help="task type")
    parser.add_argument("-tag", required=False, type=str, default=None,
                        help="additional (recommended) tag for experiment folder")
    parser.add_argument('-chkpt', required=False, type=str, default=None,
                        help="checkpoint of weights to load")
    parser.add_argument('-swap', required=False, type=int, default=10,
                        help="#batches for the module to swap backbones")
    parser.add_argument('-skip', action="store_true", default=False,
                        help="whether to use skip-connections in the head")
    parser.add_argument('-backprop', action=cast_backprop, type=int, default=0,
                        help="whether to backprop through the backbone, or specify freeze level")
    parser.add_argument('-sample_n', type=int, default=None,
                        help="detectron2 on-the-fly latent sampling n_per_bin")
    return parser.parse_known_args(ext_args)

def get_task_manager(task_name, cg_dict, task_params, book_keeper, swap_num=10, skip=False, backprop=False, chkpt=None, cache_location=None, sampling=None):
    cg = cg_dict['compute graph']
    original_config = None
    if "original config" in cg_dict.keys():
        book_keeper.log("Original config detected")
        original_config = cg_dict['original config']

    if "hpe2d" in task_name:
        from model_src.multitask.skip_adapter import SkipDeconvHPEHead
        from tasks.pose_hg_3d.hg_3d_manager import HG3DManager
        from model_src.multitask.adapt_ofa_head import OFAAdaptedCGHead
        from model_src.multitask.adapt_sampling_head import AdaptedSamplingHead

        task_manager = HG3DManager(name="Head", params=task_params, log_f=book_keeper.log)
        task_head = SkipDeconvHPEHead(task_name, hw=(task_manager.opts.output_h, task_manager.opts.output_w),
                                      joints=task_manager.opts.num_output)
            
        input_dims = [task_manager.opts.input_h, task_manager.opts.input_w, 3]
        if task_manager.opts.cache_file is not None:
            task_model = AdaptedSamplingHead(base_cg=cg, task_head=task_head, input_dims=input_dims, skip=skip)
        else:
            task_model = OFAAdaptedCGHead(base_cg=cg, original_config=original_config, task_head=task_head, 
                                        input_dims=input_dims, swap_num=swap_num, backprop=backprop, skip=skip)

        task_manager.set_model(task_model)

        if chkpt is not None:
            book_keeper.load_model_checkpoint(task_head, skip_eval_perfs=True, checkpoint_file=chkpt)
    elif "detectron" in task_name:
        from tasks.detectron2.detectron2_manager import Detectron2Manager
        # Do not comment out these imports
        from tasks.detectron2.detectron2_adapter import Detectron2Adapter
        from tasks.detectron2.sem_seg_head_adapter import SemSegFPNHAdaptedHead
        from model_src.multitask.adapt_ofa_head import OFAAdaptedCGHead

        task_manager = Detectron2Manager(task_params, exp_dir=cache_location, save=True, log_f=print)

        if "resnet" in cg.name.lower():
            from model_src.multitask.fpn_adapter_base import FeaturePyramidAdapterBase
            task_head = FeaturePyramidAdapterBase(name="Head", uniform_channels=True, hw=(256, 256), add_downsample=1)
        else:
            from model_src.multitask.fpn_adapter_detectron import FeaturePyramidDetectron
            task_head = FeaturePyramidDetectron(name="Head", hw=(256, 256), add_downsample=1)

        input_dims = [1024, 1024, 3]
        if sampling is not None:
            from model_src.multitask.latent_sample.arch_sampler import ArchSample
            from model_src.multitask.latent_sample.tensor_processor import TargetProcessor
            from model_src.multitask.adapt_sampling_head import AdaptedSamplingHead

            sampler = ArchSample(family=params.family.replace("ofa_", "").lower(), n_per_bin=sampling, use_gpu=True, use_logits=False)
            latent_processor = TargetProcessor(sampler)

            task_model = AdaptedSamplingHead(base_cg=cg, task_head=task_head, input_dims=input_dims,
                                            skip=skip, latent_processor=latent_processor)

            task_manager.set_model(task_model, chkpt_file=None)
        else:
            task_model = OFAAdaptedCGHead(base_cg=cg, original_config=original_config, 
                                        task_head=task_head, input_dims=input_dims, swap_num=swap_num,
                                        backprop=backprop, skip=skip)
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

    exp_name = "_".join([task_name, "head"])

    book_keeper = BookKeeper(log_file_name=exp_name + ".txt",
                             model_name=exp_name,
                             saved_models_dir=params.saved_models_dir,
                             logs_dir=cache_location)
    book_keeper.log("Params: {}".format(params), verbose=False)

    data_manager = FamilyDataManager([params.family], log_f=book_keeper.log)
    cg_data = data_manager.load_cache_data(params.family)

    idx = np.random.choice(list(range(len(cg_data))))
    sel_cg_dict = cg_data[idx]
    book_keeper.log("Network: {}".format(sel_cg_dict['compute graph'].name))

    task_manager = get_task_manager(task_tag, sel_cg_dict, unknown_params, book_keeper, swap_num=params.swap,
                                    skip=params.skip, backprop=params.backprop, chkpt=params.chkpt,
                                    cache_location=cache_location, sampling=params.sample_n)

    metric_dict = task_manager.train(eval_test=True)
    task_manager.model.build_overall_cg()
    book_keeper.log("Metric dict after training: {}".format(metric_dict))

    if "hpe2d" in task_tag:
        book_keeper.checkpoint_model("_head.pt", None, task_manager.model.task_head, optimizer=None)
    


if __name__ == "__main__":
    _parser = prepare_global_params()
    params, unknown_params = prepare_local_params(_parser)
    main(params, unknown_params)
