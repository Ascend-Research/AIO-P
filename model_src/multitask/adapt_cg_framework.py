import torch as t
import tensorflow.compat.v1 as tf
from model_src.comp_graph.tf_comp_graph import ComputeGraph, OP2I, NODE_TYPE2IDX
from model_src.comp_graph.tf_comp_graph_utils import get_topo_sorted_nodes, post_prune_dilation
from utils.graph_utils import get_reverse_adj_dict, get_index_based_input_inds
from model_src.predictor.gpi_family_data_manager import get_domain_configs
from model_src.comp_graph.tf_comp_graph_output import CompGraphOutputNet as TFNet
from model_src.comp_graph.tf_comp_graph_output_torch import CompGraphOutputNet as TorchNet

class TaskAdaptedCGModel(t.nn.Module):
    def __init__(self, base_cg, task_head, input_dims=[32, 32, 3]):
        super(TaskAdaptedCGModel, self).__init__()

        self.base_cg = post_prune_dilation(base_cg, keep_dil_info=True)
        self.task_head = task_head
        self.input_dims = input_dims
        self.op2i = OP2I().build_from_file()

        self._compile_network()

    def set_optim_params(self, whatever=None):
        return


    def _calc_nodes_edges(self):
        self.adapted_cg_body = post_prune_dilation(self._adapt_input_res(), keep_dil_info=True)
        self.final_res = self.adapted_cg_body.regular_nodes[-1].resolution[::2]
        nodes = self.adapted_cg_body.nodes
        src2dst_ids = self.adapted_cg_body.src_id2dst_ids_dict

        self.nodes = get_topo_sorted_nodes(nodes, get_reverse_adj_dict(src2dst_ids))
        self.node_input_inds = get_index_based_input_inds([n.str_id for n in self.nodes], src2dst_ids)

    def _compile_network(self):

        self._calc_nodes_edges()

        self.body = TorchNet(self.op2i, topo_nodes=self.nodes, net_input_inds=self.node_input_inds,
                             squeeze_output=False)

        self.task_head.build(resolution=self.final_res)

    def build_overall_cg(self):
        overall_cg_name = "_".join([self._extract_cg_name(), self.task_head.name])
        self.overall_cg = ComputeGraph(name=overall_cg_name, H=self.input_dims[0], W=self.input_dims[1],
                                       C_in=self.input_dims[2])
        self.overall_cg.build_from_model_maker(self.tf_model_maker, self.op2i, oov_threshold=0.)

    def build_network_from_cg(self, tf=False):

        new_nodes = self.overall_cg.nodes
        src2dst_ids = self.overall_cg.src_id2dst_ids_dict

        new_nodes = get_topo_sorted_nodes(new_nodes, get_reverse_adj_dict(src2dst_ids))
        node_input_inds = get_index_based_input_inds([n.str_id for n in new_nodes], src2dst_ids)

        if tf:
            return TFNet(self.op2i, topo_nodes=new_nodes, net_input_inds=node_input_inds, squeeze_output=False)
        return TorchNet(self.op2i, topo_nodes=new_nodes, net_input_inds=node_input_inds, squeeze_output=False)

    def forward(self, x):
        x = self.body(x)
        return self.task_head(x)

    def tf_model_maker(self):
        tf_body = TFNet(self.op2i, topo_nodes=self.nodes, net_input_inds=self.node_input_inds,
                        squeeze_output=False)
        tf_task_head = self.task_head.tf_model_maker()
        return TensorFlowTaskModel(tf_body, tf_task_head)

    def _extract_cg_name(self):
        return self.base_cg.name

    def _adapt_input_res(self):
        cg = self.base_cg
        [target_h, target_w, target_c] = self.input_dims

        domain_configs = get_domain_configs()

        nodes = cg.nodes
        src2dst_ids = cg.src_id2dst_ids_dict
        nodes = get_topo_sorted_nodes(nodes, get_reverse_adj_dict(src2dst_ids))
        tgt_node = None
        for n in nodes:
            if n.type_idx == NODE_TYPE2IDX["weighted"]:
                tgt_node = n
                break
        assert tgt_node is not None, "Unable to locate an input weighted node"
        Hin, Hout, Win, Wout, _, Cout = tgt_node.resolution
        _, Co, k1, k2 = tgt_node.shape
        tgt_node.resolution = [Hin, Hout, Win, Wout, target_c, Cout]
        tgt_node.shape = [target_c, Co, k1, k2]

        nodes = cg.nodes
        src2dst_ids = cg.src_id2dst_ids_dict
        nodes = get_topo_sorted_nodes(nodes, get_reverse_adj_dict(src2dst_ids))
        node_input_inds = get_index_based_input_inds([n.str_id for n in nodes], src2dst_ids)

        last_node = None
        for i, node in enumerate(nodes[::-1]):
            if node.resolution[1] > 1 and node.resolution[3] > 1:
                last_node = node
                last_node_i = len(nodes) - i
                break
        assert last_node is not None, "Unable to locate a last weighted node"

        nodes = nodes[:last_node_i]
        node_input_inds = node_input_inds[:last_node_i]

        def model_maker():
            _model = TFNet(op2i=self.op2i, name=cg.name, squeeze_output=False,
                                        topo_nodes=nodes, net_input_inds=node_input_inds)
            return lambda _x, training: _model.call(_x, training=training)

        new_cg = ComputeGraph(C_in=target_c,
                              H=target_h, W=target_w,
                              name="{}_body".format(self._extract_cg_name()),
                              max_hidden_size=domain_configs["max_hidden_size"],
                              max_kernel_size=domain_configs["max_kernel_size"],
                              max_derived_H=domain_configs["max_h"],
                              max_derived_W=domain_configs["max_w"])
        new_cg = post_prune_dilation(new_cg, keep_dil_info=True)
        new_cg.build_from_model_maker(model_maker=model_maker,
                                      op2idx=self.op2i, oov_threshold=0.)

        return new_cg


class TensorFlowTaskModel(tf.keras.Model):

    def __init__(self, body, head):
        super(TensorFlowTaskModel, self).__init__()
        self.body = body
        self.head = head

    def call(self, x, training=True):
        with tf.name_scope(self._name):
            body_output = self.body(x, training=training)
            if type(body_output) is not list:
                body_output = [body_output]
            return self.head(body_output, training=training)
