import copy
import hashlib
import collections
import tensorflow as tf
from model_src.comp_graph.tf_comp_graph_output import CompGraphOutputNet
from model_src.comp_graph.tf_comp_graph import ComputeGraph, WeightedNode, remove_node_edges
from utils.graph_utils import topo_sort_dfs, get_reverse_adj_dict, adj_dict_to_edge_list, \
    get_index_based_input_inds, edge_list_to_edge_pairs


def get_topo_sorted_nodes(nodes, src2dst_ids):
    """
    Note: sorted nodes will be the most dependent node first
    """
    new_src2dst_ids = collections.defaultdict(set)
    for k, v in src2dst_ids.items():
        new_src2dst_ids[k] = copy.deepcopy(v)
    id2node = {n.str_id:n for n in nodes}
    sorted_ids = topo_sort_dfs([n.str_id for n in nodes], new_src2dst_ids)
    sorted_nodes = [id2node[_id] for _id in sorted_ids]
    assert len(sorted_nodes) == len(nodes)
    return sorted_nodes


def post_prune_dilation(cg:ComputeGraph,
                        keywords=("spacetobatchnd", "batchtospacend"),
                        keep_dil_info=False):
    """
    Merge the 3-op group for dil convs
    This involves dropping the spacetobatchnd and batchtospacend nodes
    And also, conv ops between them will have padding reset to same
    Returns a copy
    """
    cg = copy.deepcopy(cg)
    nodes = cg.nodes
    src_id2dst_ids = cg.src_id2dst_ids_dict
    dst_id2src_ids = get_reverse_adj_dict(src_id2dst_ids)
    id2node = {n.str_id: n for n in nodes}

    def _find_next_node_rep(_nid):
        neighbor_ids = list(src_id2dst_ids[_nid])
        if len(neighbor_ids) == 1:
            return id2node[neighbor_ids[0]]
        assert False, "Cannot find a rep node for spacetobatchnd node"

    def _find_prev_node_rep(_nid):
        neighbor_ids = list(dst_id2src_ids[_nid])
        if len(neighbor_ids) == 1:
            return id2node[neighbor_ids[0]]
        assert False, "Cannot find a rep node for batchtospacend node"

    pruned_ids = set()
    for n in nodes:
        if n.label not in keywords:
            continue
        if n.label == "spacetobatchnd":
            # Find the rep_node, it has to be a weighted conv node
            rep_node = _find_next_node_rep(n.str_id)
            assert isinstance(rep_node, WeightedNode)
            if rep_node.metadata is None: rep_node.metadata = {}
            rep_node.metadata["padding"] = "same"
            rep_node.resolution = list(n.resolution)
        if n.label == "batchtospacend":
            # Find the rep_node, it has to be a weighted conv node
            rep_node = _find_prev_node_rep(n.str_id)
            assert isinstance(rep_node, WeightedNode)
            if rep_node.metadata is None: rep_node.metadata = {}
            rep_node.metadata["padding"] = "same"
            Hin, Hout, _, Wout, _, Cout = n.resolution
            rep_node.resolution[1] = Hout
            rep_node.resolution[3] = Wout
            rep_node.resolution[5] = Cout
            rep_node.resolution = tuple(rep_node.resolution)
            if keep_dil_info:
                rep_node.metadata["dil_rate"] = Hout // Hin
        # Prune n
        pruned_ids.add(n.str_id)
        remove_node_edges(n, src_id2dst_ids, dst_id2src_ids)
    kept_nodes = [n for n in nodes if n.str_id not in pruned_ids]
    cg.set_nodes_edge_pairs(kept_nodes, src_id2dst_ids)
    return cg


def get_simple_cg_str_id(cg:ComputeGraph, use_hash=True):
    """
    A quick and naive way to id a compute graph
    There is a limit on this id's ability to tell apart two CGs
    This limit should be at least as good as the limit of the gnn features to distinguish two CGs
    """
    nodes = cg.nodes
    src2dst_ids = cg.src_id2dst_ids_dict
    dst2src_ids = get_reverse_adj_dict(src2dst_ids)
    nodes = get_topo_sorted_nodes(nodes, dst2src_ids)
    edge_list = adj_dict_to_edge_list(src2dst_ids)
    edge_pairs = edge_list_to_edge_pairs(edge_list)
    cg_node_ids = []
    for ni, node in enumerate(nodes):
        if isinstance(node, WeightedNode):
            node_id = "<op{}res[{}]shape[{}]strides[{}]use_bias[{}]>".format(
                node.op_type_idx,
                ",".join([str(v) for v in node.resolution]),
                ",".join([str(v) for v in node.shape]),
                str(node.strides),
                str(node.metadata["use_bias"]) if node.metadata is not None and "use_bias" in node.metadata else "None")
        else:
            node_id = "<op{}res[{}]strides[{}]>".format(
                node.op_type_idx,
                ",".join([str(v) for v in node.resolution]),
                str(node.strides))
        cg_node_ids.append(node_id)
    cg_node_ids.sort()
    _id = "#".join(cg_node_ids) + "Edges:[{}]".format(edge_pairs)
    if use_hash:
        _id = hashlib.sha512(_id.encode("UTF-8")).hexdigest()
    return _id


def get_flops_from_cg_output_net(op2i, input_shape, nodes, input_inds,
                                 squeeze_output=True):
    def model_maker():
        _model = CompGraphOutputNet(op2i=op2i, name="", squeeze_output=squeeze_output,
                                    topo_nodes=nodes, net_input_inds=input_inds)
        return lambda _x, training: _model.call(_x, training=training)

    g = tf.Graph()
    run_meta = tf.RunMetadata()
    with g.as_default():
        sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.25),
                                                log_device_placement=False))
        batch = tf.ones(input_shape, tf.float32)
        x = tf.identity(batch, "input")
        model = model_maker()
        output = model(x, training=False)
        if isinstance(output, list):
            output_names = []
            for i, out in enumerate(output):
                out_n = tf.identity(out, "output{}".format(i))
        else:
            output = tf.identity(output, "output")
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        opts["output"] = "none"
        flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
        flops = flops.total_float_ops if flops is not None else 0
    return flops


_CG_FLOPS_MEMO = {}


def compute_cg_flops(cg:ComputeGraph, op2i,
                     use_fast_counter=True,
                     div=1e6):
    _id = get_simple_cg_str_id(cg)
    if _id in _CG_FLOPS_MEMO:
        return _CG_FLOPS_MEMO[_id]

    if use_fast_counter:
        # Use fast approximation designed for CG
        from model_src.comp_graph.tf_comp_graph_flops_counter import get_flops
        nodes = cg.nodes
        src2dst_ids = cg.src_id2dst_ids_dict
        nodes = get_topo_sorted_nodes(nodes, get_reverse_adj_dict(src2dst_ids))
        node_input_inds = get_index_based_input_inds([n.str_id for n in nodes],
                                                     src2dst_ids)
        net_flops = get_flops(op2i, nodes, node_input_inds,
                              batch_size=1, divisor=div,
                              log_f=lambda _m: _m)
    else:
        # Use TF native counter, much slower but can support all ops
        nodes = cg.nodes
        src2dst_ids = cg.src_id2dst_ids_dict
        nodes = get_topo_sorted_nodes(nodes, get_reverse_adj_dict(src2dst_ids))
        node_input_inds = get_index_based_input_inds([n.str_id for n in nodes],
                                                     src2dst_ids)
        net_flops = get_flops_from_cg_output_net(op2i, cg.input_shape,
                                                 nodes, node_input_inds)
        net_flops /= div

    _CG_FLOPS_MEMO[_id] = net_flops
    return net_flops


