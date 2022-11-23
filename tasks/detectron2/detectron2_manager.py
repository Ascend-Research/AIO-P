#!/usr/bin/env python
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

# Now a subclass of BaseTaskManager for using detectron2

from collections import OrderedDict
import os
import sys
import torch
from torch.nn.parallel import DistributedDataParallel
import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.utils.events import CommonMetricPrinter
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from tasks.task_manager_base import BaseTaskManager
from detectron2.config import CfgNode as CN
from params import P_SEP
from multiprocessing import Pipe
import numpy as np
from collections import Counter
from fvcore.nn import flop_count_table  # can also try flop_count_str
from detectron2.config import CfgNode, get_cfg, instantiate
from detectron2.data import build_detection_test_loader
from detectron2.engine import default_argument_parser
from detectron2.modeling import build_model
from detectron2.utils.analysis import FlopCountAnalysis, parameter_count

# Don't remove these imports!
from tasks.detectron2.sem_seg_head_adapter import SemSegFPNHAdaptedHead
from tasks.detectron2.detectron2_adapter import Detectron2Adapter

class Detectron2Manager(BaseTaskManager):

    def __init__(self, args, exp_dir=None, save=False, log_f=print):
        super(Detectron2Manager, self).__init__(log_f=log_f)

        # https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/defaults.py#L126
        # Detectron2 uses the process user ID (os.getuid()) which is always the same. Downside is that 
        # running multiple experiments with the distributed framework doesn't work because they're 
        # all clamoring for the same port, so only one can run at a time.
        # Therefore, we copy their code but use os.getpid(), the process ID instead
        # For the concern about orphan processes, we can see what to 'sudo kill -9' by looking at 'nvidia-smi'
        port = 2 ** 15 + 2 ** 14 + hash(os.getpid() if sys.platform != "win32" else 1) % 2 ** 14
        port_arg_list = ['--dist-url', 'tcp://127.0.0.1:{}'.format(port)]
        args = port_arg_list + args

        self.args = default_argument_parser().parse_args(args)
        self.cfg = get_cfg()
        self.cfg.merge_from_file(self.args.config_file)
        self.cfg.merge_from_list(self.args.opts)
        self.cfg.OUTPUT_DIR = exp_dir
        self.save = save

    # This is the public-facing set_model function
    # The full model and its optimizer are instantiated by _train for DistributedDataParallel
    def set_model(self, model, chkpt_file=None, chkpt_strict=True):
        self.model = model
        self.cfg.MODEL.WEIGHTS = chkpt_file
        self.cfg.MODEL.STRICT = chkpt_strict

    # Should only be called after training
    def build_cgs(self, return_dict=False):
        self.model_adapter.backbone.build_overall_cg(self.cfg)
        self.cfg.freeze()

        if return_dict:
            return self.model_adapter.backbone.overall_cg
        
    def eval(self):
        raise NotImplementedError(
            "Use train(eval_test=True) with a properly configured .yml file for detectron2")
        
    def train(self, eval_test=True):
        dist_info = "Distributed Learning Params:\n"
        dist_info += "num_gpus: %d;\n" % self.args.num_gpus
        dist_info += "num_machines: %d;\n" % self.args.num_machines
        dist_info += "machine_rank: %d;\n" % self.args.machine_rank
        dist_info += "dist_url: %s;\n" % self.args.dist_url
        self.log_f(dist_info)

        pipe_output, pipe_input = Pipe(duplex=False)

        launch(
            _train,
            self.args.num_gpus,
            num_machines=self.args.num_machines,
            machine_rank=self.args.machine_rank,
            dist_url=self.args.dist_url,
            args=(self.model, self.cfg, self.args, self.log_f, pipe_input, eval_test, self.save),
        )

        [self.train_dict, self.test_dict] = pipe_output.recv()

        self.cfg.defrost()
        self.model_adapter, self.config = make_adapted_model(self.model, self.cfg)
        return {**self.train_dict, **self.test_dict}
    
    def profiler(self, profiler_metrics,  num_inputs=3, log_f=print):
        """
        Calculates the FLOPS and num of model parameters
        FLOPS calculation from here: https://github.com/facebookresearch/detectron2/blob/main/tools/analyze_model.py#L40
        Model parameter calculation from here: https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/print_model_statistics.py#L626
        - num_inputs: the number of images to average over
        """
        profiler_output = {}

        import copy
        cfg = copy.deepcopy(self.cfg)
        model = copy.deepcopy(self.model)

        if "flops" in profiler_metrics or "params" in profiler_metrics:
            if not hasattr(self, "profiler_imgs"):
                cfg = setup(cfg, self.args)
                if isinstance(cfg, CfgNode):
                    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=None)
                else:
                    data_loader = instantiate(cfg.dataloader.test)
                self.profiler_imgs = [img for img in zip(range(num_inputs), data_loader)] 

            model, cfg = make_adapted_model(model, cfg)
            if cfg.is_frozen(): cfg.defrost()
            model.eval()

            counts = Counter()
            total_flops = []
            for idx, data in self.profiler_imgs:
                flops = FlopCountAnalysis(model, data)
                if idx > 0:
                    flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
                counts += flops.by_operator()
                total_flops.append(flops.total())

            if "flops" in profiler_metrics:                
                OBJ_DET_FLOPS_KEYS = ["model.backbone.adapted_model", "model.proposal_generator.rpn_head", "model.roi_heads.box_head", "model.roi_heads.box_predictor"]
                INST_SEG_FLOPS_KEYS = ["model.backbone.adapted_model", "model.proposal_generator.rpn_head", "model.roi_heads.mask_head"]
                SEM_SEG_FLOPS_KEYS = ["model.backbone.adapted_model", "model.sem_seg_head"]
                PAN_SEG_FLOPS_KEYS = ["model.backbone.adapted_model", "model.proposal_generator.rpn_head", "model.roi_heads.box_head", "model.roi_heads.box_predictor", 
                                                         "model.roi_heads.mask_head", "model.sem_seg_head"]

                log_f("Flops table computed from only one input sample:\n" + flop_count_table(flops))
                log_f("Average GFlops for each type of operators:\n"+ str([(k, v / (idx + 1) / 1e9) for k, v in counts.items()]))
                log_f("Total GFlops: {:.1f}±{:.1f}".format(np.mean(total_flops) / 1e9, np.std(total_flops) / 1e9))
                
                # FLOPS by task
                divisor = 1e9 # billion (G)
                flop_modules = flops.by_module()

                # Doing override for mbv3 because roi.head for flops is 0
                flop_modules["model.roi_heads.box_head"] = 13.894 * divisor
                flop_modules["model.roi_heads.box_predictor"] = 0.411 * divisor
                flop_modules["model.roi_heads.mask_head"] = 52.986 * divisor

                flops_pan_seg = sum([flop_modules[m] for m in PAN_SEG_FLOPS_KEYS]) / divisor
                flops_obj_det = sum([flop_modules[m] for m in OBJ_DET_FLOPS_KEYS]) / divisor
                flops_inst_seg = sum([flop_modules[m] for m in INST_SEG_FLOPS_KEYS]) / divisor
                flops_sem_seg = sum([flop_modules[m] for m in SEM_SEG_FLOPS_KEYS]) / divisor
                    
                profiler_output["flops_pan_seg"] = flops_pan_seg
                profiler_output["flops_obj_det"] = flops_obj_det
                profiler_output["flops_inst_seg"] = flops_inst_seg
                profiler_output["flops_sem_seg"] = flops_sem_seg

            # Parameters 
            if "params" in profiler_metrics:
                BASE_PARAM_KEYS = ["model.backbone.adapted_model"]
                OBJ_DET_PARAMS_KEYS = BASE_PARAM_KEYS + ["model.proposal_generator.rpn_head", "model.roi_heads.box_head", "model.roi_heads.box_predictor"]
                INST_SEG_PARAMS_KEYS = BASE_PARAM_KEYS + ["model.proposal_generator.rpn_head", "model.roi_heads.mask_head"]
                SEM_SEG_PARAMS_KEYS = BASE_PARAM_KEYS + ["model.sem_seg_head"]
                PAN_SEG_PARAMS_KEYS = BASE_PARAM_KEYS + ["model.proposal_generator.rpn_head", "model.roi_heads.box_head", "model.roi_heads.box_predictor", 
                                                         "model.roi_heads.mask_head", "model.sem_seg_head"]

                divisor = 1e6 # million (M)
                model_params = dict(parameter_count(flops._model))
                params_pan_seg = sum([model_params[m] for m in PAN_SEG_PARAMS_KEYS]) / divisor
                params_obj_det = sum([model_params[m] for m in OBJ_DET_PARAMS_KEYS]) / divisor
                params_inst_seg = sum([model_params[m] for m in INST_SEG_PARAMS_KEYS]) / divisor
                params_sem_seg = sum([model_params[m] for m in SEM_SEG_PARAMS_KEYS]) / divisor
                
                profiler_output["params_pan_seg"] = params_pan_seg
                profiler_output["params_obj_det"] = params_obj_det
                profiler_output["params_inst_seg"] = params_inst_seg
                profiler_output["params_sem_seg"] = params_sem_seg

        return profiler_output


def _train(model, cfg, args, log_f, pipe, eval_test=False, save=False):

    cfg = setup(cfg, args)

    model, cfg = make_adapted_model(model, cfg)
    if comm.is_main_process():
        log_f("Model:\n{}".format(model))
    backbone = model.backbone

    if cfg.MODEL.WEIGHTS is not None:
        load_model(model, cfg)

    if comm.get_world_size() > 1:
        model = DistributedDataParallel(model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=True)

    model.train()

    start_iter = 0
    max_iter = cfg.SOLVER.MAX_ITER

    if max_iter > 0:
        optimizer = build_optimizer(cfg, model)
        backbone.adapted_model.optim = optimizer
        backbone.adapted_model.set_optim_params()
        scheduler = build_lr_scheduler(cfg, optimizer)

    writers = [CommonMetricPrinter(max_iter)] if comm.is_main_process() else []

    data_loader = build_detection_train_loader(cfg)

    if comm.is_main_process():
        log_f("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        loss_dict_reduced = {}
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1):
                for writer in writers:
                    writer.write()

            # Save every 1000 steps
            if comm.is_main_process() and save and iteration - start_iter > 5 and (iteration + 1) % 1000 == 0:
                save_model(model, cfg)
        
        test_dict = eval(cfg, model, log_f) if eval_test else {}
        if comm.is_main_process():
            if save:
                save_model(model, cfg)
            pipe.send([loss_dict_reduced, test_dict])
        comm.synchronize()


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

def eval(cfg, model, log_f):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(cfg, dataset_name=dataset_name, 
                                  output_folder=None)
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            log_f("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    
    return results


def make_adapted_model(model, cfg):

    cfg.set_new_allowed(True)
    cfg.MODEL.COMPUTE_GRAPH = CN()
    cfg.MODEL.COMPUTE_GRAPH.ADAPTED_MODEL = [model]
    model = build_model(cfg)
    cfg.set_new_allowed(False)
    cfg.freeze() 

    return model, cfg


def save_model(model, cfg):
    if isinstance(model, DistributedDataParallel):
        model = model.module

    save_dict = {'upsampler': model.backbone.adapted_model.task_head.state_dict()}

    meta_arch = cfg.MODEL.META_ARCHITECTURE
    if meta_arch == "GeneralizedRCNN" or meta_arch == "PanopticFPN":
        save_dict['roi_heads'] = model.roi_heads.state_dict()
        if cfg.MODEL.RPN is not None:
            save_dict['proposal_generator'] = model.proposal_generator.state_dict()

    if meta_arch == "PanopticFPN":
        save_dict['sem_seg_head'] = model.sem_seg_head.state_dict()

    save_file_loc = P_SEP.join([cfg.OUTPUT_DIR, "head_weights.pkl"])
    torch.save(save_dict, save_file_loc)

def load_model(model, cfg):
    map_loc = 'cuda:{}'.format(comm.get_local_rank())
    print("Loading from...")
    print(cfg.MODEL.WEIGHTS)
    save_dict = torch.load(cfg.MODEL.WEIGHTS, map_location=map_loc)

    model.backbone.adapted_model.task_head.load_state_dict(save_dict['upsampler'], strict=cfg.MODEL.STRICT)

    meta_arch = cfg.MODEL.META_ARCHITECTURE
    if meta_arch == "GeneralizedRCNN" or meta_arch == "PanopticFPN":
        model.roi_heads.load_state_dict(save_dict['roi_heads'])
        if cfg.MODEL.RPN is not None:
            model.proposal_generator.load_state_dict(save_dict['proposal_generator'])

    if meta_arch == "PanopticFPN":
        model.sem_seg_head.load_state_dict(save_dict['sem_seg_head'])


def setup(cfg, args):
    """
    Create configs and perform basic setups.
    """
    default_setup(
        cfg, args
    )
    return cfg

