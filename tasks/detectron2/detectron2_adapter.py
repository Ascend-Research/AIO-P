from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from math import log2
from detectron2.layers import ShapeSpec
from model_src.multitask.adapt_cg_framework import TensorFlowTaskModel
from detectron2.config import get_cfg

class Detectron2Adapter(Backbone):

    def __init__(self, adapted_model):
        super(Detectron2Adapter, self).__init__()
        self.adapted_model = adapted_model
        self._out_feature_channels, self._out_feature_strides = {}, {}
        self._out_features = []
        self._calc_output_shape()
        size_div_out_feat = self._out_features[self.adapted_model.task_head.add_downsample]
        self._size_divisibility = self._out_feature_strides[size_div_out_feat]
        self.makers = {
            "obj_det": self.tf_faster_rcnn_maker,
            "inst_seg": self.tf_inst_seg_maker,
            "sem_seg": self.tf_sem_seg_maker,
            "pan_seg": self.tf_pan_seg_maker,
        }

    def _calc_output_shape(self):
        input_res = self.adapted_model.input_dims[0]
        intermediate_res = self.adapted_model.task_head.input_res[0]
        final_res = self.adapted_model.task_head.hw[0]
        p_max = int(log2(input_res / intermediate_res))
        if not self.adapted_model.skip:
            p_min = p_max
        else:
            p_min = int(log2(input_res / final_res))

        for i in range(p_min, p_max + self.adapted_model.task_head.add_downsample + 1):
            self._out_features.append("p%d" % i)
            self._out_feature_channels[self._out_features[-1]] = 256
            self._out_feature_strides[self._out_features[-1]] = 2 ** i
        self._out_features.reverse()

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        x = self.adapted_model(x)
        output_dict = {}
        for i, key in enumerate(self._out_features):
            output_dict[key] = x[i]
        return output_dict

    def build_overall_cg(self, cfg):
        from model_src.comp_graph.tf_comp_graph import ComputeGraph
        
        if isinstance(cfg, str):
            self.cfg = get_cfg()
            self.cfg.merge_from_file(cfg)
        else:
            self.cfg = cfg
        self.overall_cg = {}
        for task in self.determine_tasks():
            cg_name = "_".join([self.adapted_model._extract_cg_name(), task])
            task_cg = ComputeGraph(name=cg_name, H=self.adapted_model.input_dims[0],
            W = self.adapted_model.input_dims[1], C_in=self.adapted_model.input_dims[2])
            task_cg.build_from_model_maker(self.makers[task], self.adapted_model.op2i, oov_threshold=0.)
            self.overall_cg[task] = task_cg

    def determine_tasks(self):
        if self.cfg.MODEL.META_ARCHITECTURE == "PanopticFPN":
            return self.makers.keys() 
        if self.cfg.MODEL.MASK_ON:
            return ["obj_det", "inst_seg"] 
        else:
            return ["obj_det"]

    def tf_faster_rcnn_maker(self):
        from tasks.detectron2.faster_rcnn_tf import TFFasterRCNN
        tf_body = self.adapted_model.tf_model_maker()
        tf_task_head = TFFasterRCNN()
        return TensorFlowTaskModel(tf_body, tf_task_head)

    def tf_inst_seg_maker(self):
        from tasks.detectron2.mask_rcnn_tf import TFMaskRCNN
        tf_body = self.adapted_model.tf_model_maker()
        tf_task_head = TFMaskRCNN()
        return TensorFlowTaskModel(tf_body, tf_task_head)

    def tf_sem_seg_maker(self):
        from tasks.detectron2.sem_seg_tf import TFSemSeg
        tf_body = self.adapted_model.tf_model_maker()
        tf_task_head = TFSemSeg()
        return TensorFlowTaskModel(tf_body, tf_task_head)

    def tf_pan_seg_maker(self):
        from tasks.detectron2.sem_seg_tf import TFPanSeg
        tf_body = self.adapted_model.tf_model_maker()
        tf_task_head = TFPanSeg()
        return TensorFlowTaskModel(tf_body, tf_task_head)


@BACKBONE_REGISTRY.register()
def build_detectron2_adapter(cfg, input_shape: ShapeSpec):

    model = Detectron2Adapter(adapted_model=cfg.MODEL.COMPUTE_GRAPH.ADAPTED_MODEL[0])

    return model
