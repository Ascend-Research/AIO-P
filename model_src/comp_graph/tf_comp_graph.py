import time
import copy
import warnings
import collections
from abc import ABC
from params import *
from constants import OOV_TOKEN
import tensorflow.compat.v1 as tf
from utils.misc_utils import RunningStatMeter
from tensorflow.python.framework import graph_util, tensor_util
from utils.graph_utils import edge_pairs_to_edge_list, get_reverse_adj_dict
from google.protobuf.json_format import MessageToDict


NODE_TYPE2IDX = {
    "regular": 0,
    "weighted": 1,
}
SHAPE_VEC_DIM = 4 # (input size, output size, kernel 1, kernel 2)
TYPE_VEC_DIM = max(NODE_TYPE2IDX.values()) + 1
WEIGHTED_NODE_KEYWORDS = {
    "kernel", "conv", "depthwise", "matmul", "biasadd", "pointwise", "weight",
}

KEEP_NODE_KEYWORDS = {
    "input", "output",
    "kernel", "conv2d",
    "add", "relu", "relu6", "fusedbatchnorm",
    "matmul", "depthwise", "depthwise_kernel",
    "pointwise", "weight",
    "sigmoid", "tanh", "swish", "pool", "mul",
    "spacetobatchnd", "batchtospacend",
    "spacetodepthnd", "depthtospacend",
    "mean", "global",
    "identity", "zero", "concat", "gelu", "paddings", "resizebilinear"
    "reserved",
}
EXCLUDE_NODE_KEYWORDS = {
    "moving_mean",
}
C_CHANGE_CAPABLE_NODE_TYPES = {
    # Should match the definition in tf_primitive_ops_simplified.txt
    "conv",
}
KEEP_NODE_OP_KEYWORDS = {
    # For those nodes with custom names but is actuall a known type
    "sigmoid", "relu", "tanh", "swish",
}


def _resize_graph(dot, size_per_element=0.5, min_size=12):
    # Get the approximate number of nodes and edges
    num_rows = len(dot.body)
    content_size = num_rows * size_per_element
    size = max(min_size, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)


class OP2I:

    def __init__(self, oov_token=OOV_TOKEN, allow_overlap_mapping=True):
        super(OP2I, self).__init__()
        self.oov_token = oov_token.lower()
        self.oov_idx = 0
        self.allow_overlap_mapping = allow_overlap_mapping
        self.op_labels = ()
        self._base_op2idx = {}
        self._base_idx2op = {}
        self._cached_op2idx = {}

    def __str__(self):
        return "OP2I[base_vocab_size={}, cached_vocab_size={}]".format(self.base_vocab_size, self.cached_vocab_size)

    def __repr__(self):
        return str(self)

    def contains_op(self, op):
        if op == self.oov_token: return True
        return self._query_idx(op) != self.oov_idx

    @property
    def base_vocab_size(self):
        return len(self._base_op2idx)

    @property
    def cached_vocab_size(self):
        return len(self._cached_op2idx)

    def build_from_file(self, src_file=P_SEP.join([DATA_DIR, "tf_primitive_ops_simplified.txt"])):
        self._base_op2idx = {}
        self._base_idx2op = {}
        self._cached_op2idx = {}
        op_labels = []
        with open(src_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0: continue
            op_labels.append(line.lower())
        if self.oov_token in op_labels:
            op_labels.remove(self.oov_token)
        self.op_labels = tuple([self.oov_token] + op_labels)
        for i, op in enumerate(self.op_labels):
            self._base_idx2op[i] = op
            self._base_op2idx[op] = i
        assert len(self._base_op2idx) == len(self._base_idx2op)
        return self

    def _query_idx(self, op):
        op = op.lower()
        if op in self._base_op2idx:
            return self._base_op2idx[op]
        if op in self._cached_op2idx:
            return self._cached_op2idx[op]
        if self.allow_overlap_mapping:
            cand_base_ops = []
            for base_op in self.op_labels:
                if base_op.lower() in op.lower():
                    cand_base_ops.append(base_op)
            if len(cand_base_ops) > 0:
                cand_base_ops.sort(key=lambda s:len(s), reverse=True)
                sel_op = cand_base_ops[0]
                self._cached_op2idx[op] = self._base_op2idx[sel_op]
                return self._base_op2idx[sel_op]
        return self.oov_idx

    def query_op(self, idx):
        return self._base_idx2op[idx]

    def __getitem__(self, op):
        rv = self._query_idx(op)
        return rv

    def __contains__(self, item):
        return self.contains_op(item)

    def __len__(self):
        return len(self._base_op2idx)


class Node(ABC):

    def __init__(self, str_id, label, type_idx):
        self.str_id = str_id
        self.type_idx = type_idx
        self.label = label
        self.resolution = None # Derived [Hin, Hout, Win, Wout, Cin, Cout] for every node
        self.strides = None # Mainly used for deriving resolution, [NHWC] format, first and last value should be 1
        self.metadata = None # Other node specific metadata

    def __str__(self):
        return self.label + "[id={}]".format(self.str_id)

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memodict={}):
        node = Node(self.str_id, self.label, self.type_idx)
        node.resolution = copy.deepcopy(self.resolution)
        node.strides = copy.deepcopy(self.strides)
        node.metadata = copy.deepcopy(self.metadata)
        return node


class RegularNode(Node):

    def __init__(self, str_id, label, op_type_idx,
                 type_idx=NODE_TYPE2IDX["regular"]):
        super(RegularNode, self).__init__(str_id, label, type_idx)
        self.op_type_idx = op_type_idx

    def __str__(self):
        return self.label + "[op_type_idx={}\nres={}]".format(self.op_type_idx, self.resolution)

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memodict={}):
        node = RegularNode(self.str_id, self.label, self.op_type_idx,
                           type_idx=self.type_idx)
        node.resolution = copy.deepcopy(self.resolution)
        node.strides = copy.deepcopy(self.strides)
        node.metadata = copy.deepcopy(self.metadata)
        return node


class WeightedNode(Node):

    def __init__(self, str_id, label, op_type_idx, shape,
                 type_idx=NODE_TYPE2IDX["weighted"]):
        super(WeightedNode, self).__init__(str_id, label, type_idx)
        self.op_type_idx = op_type_idx
        self.shape = shape # Extracted [Ci, Co, k1, k2]

    def __str__(self):
        rounded_shape = [round(v, 2) for v in self.shape]
        return self.label + "[op_type_idx={}\nshape={}\nres={}]".format(self.op_type_idx,
                                                                        rounded_shape, self.resolution)

    def __repr__(self):
        return str(self)

    def __deepcopy__(self, memodict={}):
        node = WeightedNode(self.str_id, self.label, self.op_type_idx, copy.deepcopy(self.shape),
                            type_idx=self.type_idx)
        node.resolution = copy.deepcopy(self.resolution)
        node.strides = copy.deepcopy(self.strides)
        node.metadata = copy.deepcopy(self.metadata)
        return node


def find_nodes_bfs(graph_def, op2idx):
    nodes, dst_id2src_ids = [], collections.defaultdict(set)
    curr_id = 0
    for n in graph_def.node:
        resizeFlag = False
        name = n.name.lower()
        nid = str(curr_id) + "|" + name
        label = name.split("/")[-1]
        op = n.op.lower()
        if "resizebilinear" in label:
            label = "identity"
            op = "identity"
            resizeFlag = True
        curr_id += 1
        op_type_idx = op2idx[label]
        if op_type_idx == op2idx[OOV_TOKEN] and \
            op in KEEP_NODE_OP_KEYWORDS:
            label = op
            op_type_idx = op2idx[label]
        if any(w in label for w in WEIGHTED_NODE_KEYWORDS):
            v = n.attr["value"].tensor.tensor_shape
            shape = [0, 0, 0, 0]
            if len(v.dim) == 4: # Conv kernel
                k1, k2 = v.dim[0].size, v.dim[1].size
                ci, co = v.dim[2].size, v.dim[3].size
                shape = [ci, co, k1, k2]
            elif len(v.dim) == 2: # FC
                ci, co = v.dim[0].size, v.dim[1].size
                shape = [ci, co, 0, 0]
            node = WeightedNode(nid, label, op_type_idx, shape)
        else:
            node = RegularNode(nid, label, op_type_idx)
        # Special treatment for padding nodes
        if node.label == "paddings":
            pad_tsr = tensor_util.MakeNdarray(n.attr['value'].tensor)
            assert pad_tsr.shape[0] in {2, 4}
            if pad_tsr.shape[0] == 2:
                h1, h2 = int(pad_tsr[0, 0]), int(pad_tsr[0, 1])
                w1, w2 = int(pad_tsr[1, 0]), int(pad_tsr[1, 1])
            else:
                h1, h2 = int(pad_tsr[1, 0]), int(pad_tsr[1, 1])
                w1, w2 = int(pad_tsr[2, 0]), int(pad_tsr[2, 1])
            if node.metadata is None: node.metadata = {}
            node.metadata["pad_h"] = (h1, h2)
            node.metadata["pad_w"] = (w1, w2)
        # Try to get additional data like stride
        if "strides" in n.attr:
            strides = n.attr["strides"].list.i
            node.strides = tuple(int(v) for v in strides)
        if "data_format" in n.attr:
            if node.metadata is None: node.metadata = {}
            node.metadata["data_format"] = n.attr["data_format"].s.decode("utf-8").upper()
        if "dilations" in n.attr:
            dilations = n.attr["dilations"].list.i
            if node.metadata is None: node.metadata = {}
            node.metadata["dilations"] = tuple(int(v) for v in dilations)
        if "padding" in n.attr:
            if node.metadata is None: node.metadata = {}
            s = n.attr["padding"].s.decode("utf-8").lower()
            node.metadata["padding"] = s
        if "pool" in node.label:
            if node.metadata is None: node.metadata = {}
            node.metadata["ksize"] = tuple(int(v) for v in n.attr["ksize"].list.i)
        if resizeFlag:
            if node.metadata is None: 
                node_dict = MessageToDict(n)['attr']
                node.metadata = {"align_corners": node_dict['align_corners']['b']}

        for src_name in n.input:
            src_name = src_name.lower()
            for cand_n in nodes[::-1]:
                cand_name = cand_n.str_id.split("|")[-1]
                if cand_name == src_name:
                    dst_id2src_ids[node.str_id].add(cand_n.str_id)
                    break
        nodes.append(node)
    src_id2dst_ids = collections.defaultdict(set)
    for dst_id, src_ids in dst_id2src_ids.items():
        for src_id in list(src_ids):
            src_id2dst_ids[src_id].add(dst_id)
    return nodes, src_id2dst_ids


def remove_node_edges(n, src2dst_ids, dst2src_ids):
    """
    In-place edge removal
    """
    parent_ids = dst2src_ids[n.str_id]
    children_ids = src2dst_ids[n.str_id]
    for parent_id in parent_ids:
        src2dst_ids[parent_id].remove(n.str_id)
        for child_id in children_ids:
            src2dst_ids[parent_id].add(child_id)
    for child_id in children_ids:
        dst2src_ids[child_id].remove(n.str_id)
        for parent_id in parent_ids:
            dst2src_ids[child_id].add(parent_id)
    del src2dst_ids[n.str_id]
    del dst2src_ids[n.str_id]


def prune_nodes_by_keywords(nodes, src_id2dst_ids):
    # Removes unwanted nodes by keywords
    dst_id2src_ids = get_reverse_adj_dict(src_id2dst_ids)
    pruned_ids = set()
    for n in nodes:
        if n.label in KEEP_NODE_KEYWORDS:  
            continue
        elif n.label not in EXCLUDE_NODE_KEYWORDS and \
                any(w in n.label for w in KEEP_NODE_KEYWORDS):
            continue # Partial overlap with pre-exclusion
        # Prune after this point
        pruned_ids.add(n.str_id)
        remove_node_edges(n, src_id2dst_ids, dst_id2src_ids)
    kept_nodes = [n for n in nodes if n.str_id not in pruned_ids]
    return kept_nodes, src_id2dst_ids


def merge_kernel_nodes(nodes, src_id2dst_ids):
    # Merge kernel nodes into their adjacent weight nodes
    id2node = {n.str_id: n for n in nodes}
    dst_id2src_ids = get_reverse_adj_dict(src_id2dst_ids)

    def _find_weighted_node_rep(_nid):
        for _id in list(src_id2dst_ids[_nid]):
            if isinstance(id2node[_id], WeightedNode):
                return id2node[_id]
        return None

    pruned_ids = set()
    for n in nodes:
        if not isinstance(n, WeightedNode):
             continue
        if "kernel" in n.label or "weight" in n.label:
            rep_node = _find_weighted_node_rep(n.str_id)
            assert rep_node is not None, "Cannot find an adjacent weighted node for kernel node"
            rep_node.shape = n.shape
            if n.metadata is not None:
                if rep_node.metadata is None:
                    rep_node.metadata = n.metadata
                else:
                    rep_node.metadata.update(n.metadata)
            # Prune the kernel node
            pruned_ids.add(n.str_id)
            remove_node_edges(n, src_id2dst_ids, dst_id2src_ids)
    kept_nodes = [n for n in nodes if n.str_id not in pruned_ids]
    return kept_nodes, src_id2dst_ids


def merge_bias_nodes(nodes, src_id2dst_ids):
    # Merge bias nodes into their adjacent weight nodes
    id2node = {n.str_id: n for n in nodes}
    dst_id2src_ids = get_reverse_adj_dict(src_id2dst_ids)

    def _find_node_rep(_nid):
        neighbor_ids = list(dst_id2src_ids[_nid])
        if len(neighbor_ids) == 1:
            return id2node[neighbor_ids[0]]
        elif len(neighbor_ids) > 1:
            for _id in neighbor_ids:
                if isinstance(id2node[_id], WeightedNode):
                    return id2node[_id]
        assert False, "Cannot find a rep node for bias node"

    pruned_ids = set()
    for n in nodes:
        if "bias" in n.label:
            rep_node = _find_node_rep(n.str_id)
            if rep_node.metadata is None:
                rep_node.metadata = {}
            rep_node.metadata["use_bias"] = True
            # Prune the bias node
            pruned_ids.add(n.str_id)
            remove_node_edges(n, src_id2dst_ids, dst_id2src_ids)
    kept_nodes = [n for n in nodes if n.str_id not in pruned_ids]
    return kept_nodes, src_id2dst_ids


def merge_bn_mean_nodes(nodes, src_id2dst_ids):
    # Merge any dangling mean nodes into bn
    id2node = {n.str_id: n for n in nodes}
    dst_id2src_ids = get_reverse_adj_dict(src_id2dst_ids)
    pruned_ids = set()
    for n in nodes:
        if "fusedbatchnorm" in n.label:
            parent_ids = list(dst_id2src_ids[n.str_id])
            for nid in parent_ids:
                if (nid not in dst_id2src_ids or len(dst_id2src_ids[nid]) == 0) and \
                    "mean" in id2node[nid].label:
                    # If the node as no parent and is a mean node
                    pruned_ids.add(nid)
                    remove_node_edges(id2node[nid], src_id2dst_ids, dst_id2src_ids)
    kept_nodes = [n for n in nodes if n.str_id not in pruned_ids]
    return kept_nodes, src_id2dst_ids


def re_connect_pad_nodes(nodes, src_id2dst_ids):
    # Re-connect dangling pad nodes to before their adjacent nodes
    id2node = {n.str_id: n for n in nodes}
    dst_id2src_ids = get_reverse_adj_dict(src_id2dst_ids)

    def _find_node_rep(_nid):
        assert len(src_id2dst_ids[_nid]) == 1
        for _id in list(src_id2dst_ids[_nid]):
            return id2node[_id]
        return None

    pruned_ids = set()
    for n in nodes:
        if n.label == "paddings":
            rep_node = _find_node_rep(n.str_id)
            assert rep_node is not None, "Cannot find an adjacent node for padding node"
            if "spacetobatch" in rep_node.label:
                # Special case: we do not reconnect padding nodes for the dilation component
                # Here we just prune the padding node, but add the padding info to the rep node still
                if rep_node.metadata is None: rep_node.metadata = {}
                rep_node.metadata["pad_h"] = n.metadata["pad_h"]
                rep_node.metadata["pad_w"] = n.metadata["pad_w"]
                pruned_ids.add(n.str_id)
                remove_node_edges(n, src_id2dst_ids, dst_id2src_ids)
            else:
                # Move the pad node to before the rep node
                # First find all nodes that go into the rep node
                rep_neighbors = {src_id for src_id, dst_ids in src_id2dst_ids.items()
                                 if rep_node.str_id in dst_ids and src_id != n.str_id}
                assert len(rep_neighbors) > 0
                # Then reconnect them to the pad node
                for src_id in rep_neighbors:
                    dst_ids = src_id2dst_ids[src_id]
                    dst_ids.remove(rep_node.str_id)
                    dst_ids.add(n.str_id)
                    dst_id2src_ids[n.str_id].add(src_id)
                # Remove rep_node's backward reference to these nodes
                dst_id2src_ids[rep_node.str_id] = {n.str_id}
    kept_nodes = [n for n in nodes if n.str_id not in pruned_ids]
    return kept_nodes, src_id2dst_ids


def _get_node_res(graph_op):
    if len(graph_op.inputs) == 0:
        return None # No op for input nodes
    if len(graph_op.outputs) == 0:
        return None # No op for output nodes
    if "_transpose" in graph_op.name:
        input_tsr = graph_op.inputs[-1]
    else:
        input_tsr = graph_op.inputs[0]
    output_tsr = graph_op.outputs[0]
    input_shape = [d.value for d in input_tsr.shape.dims]
    output_shape = [d.value for d in output_tsr.shape.dims]
    assert len(input_shape) == 0 or len(input_shape) == 2 or len(input_shape) == 4
    assert len(output_shape) == 2 or len(output_shape) == 4
    if len(input_shape) == 0:
        H_in, W_in, C_in = None, None, None
    elif len(input_shape) == 2:
        H_in, W_in = 1, 1
        C_in = input_shape[1]
    else:
        H_in, W_in, C_in = input_shape[1:]
    if len(output_shape) == 2:
        H_out, W_out = 1, 1
        C_out = output_shape[1]
    else:
        H_out, W_out, C_out = output_shape[1:]
    if H_in is None:
        H_in = H_out
    if W_in is None:
        W_in = W_out
    if C_in is None:
        C_in = C_out
    return H_in, H_out, W_in, W_out, C_in, C_out


def derive_node_resolution(node, id2node, src_id2dst_ids, dst_id2src_ids):
    # If we are not able to extract the res of a node
    # Call this to get the derived shape based on its prev and next nodes
    # NOTE: prev and next nodes must have resolution already set, else it'll raise an error
    parent_node = id2node[list(dst_id2src_ids[node.str_id])[0]]
    child_node = id2node[list(src_id2dst_ids[node.str_id])[0]]
    H_in = parent_node.resolution[1]
    W_in = parent_node.resolution[3]
    C_in = parent_node.resolution[5]
    H_out = child_node.resolution[0]
    W_out = child_node.resolution[2]
    C_out = child_node.resolution[4]
    return H_in, H_out, W_in, W_out, C_in, C_out


def extract_node_resolutions(start_node, nodes, src_id2dst_ids, graph_def,
                             C, H, W):
    graph_name2op = {}
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        for name, op in graph._nodes_by_name.items():
            name = name.lower()
            assert name not in graph_name2op
            graph_name2op[name] = op
    assert len(graph_name2op) > 0

    dst_id2src_ids = get_reverse_adj_dict(src_id2dst_ids)

    start_node.resolution = (H, H, W, W, C, C)
    id2node = {n.str_id: n for n in nodes}
    q = [start_node]
    visited = {start_node.str_id,}
    max_H, max_W = H, W
    unknown_node_ids = set()
    node_visit_counts = collections.defaultdict(int)

    while len(q) > 0:
        n = q.pop(0)
        assert node_visit_counts[n.str_id] < 1
        node_visit_counts[n.str_id] += 1
        name = n.str_id.split("|")[-1]
        assert name in graph_name2op, "Unknown name: {}".format(name)
        resolution = _get_node_res(graph_name2op[name])

        if resolution is None or None in resolution:
            n.resolution = None
            unknown_node_ids.add(n.str_id)
        else:
            n.resolution = resolution
            max_H = max(max_H, max(n.resolution[0], n.resolution[1]))
            max_W = max(max_W, max(n.resolution[2], n.resolution[3]))
        for dst_id in src_id2dst_ids[n.str_id]:
            if dst_id not in visited:
                new_node = id2node[dst_id]
                visited.add(new_node.str_id)
                q.append(new_node)

    assert len(visited) == len(id2node)

    input_nodes = get_input_nodes(nodes, src_id2dst_ids,
                                  check_single_input=False)
    for node in input_nodes:
        if node.resolution is None:
            node.resolution = (H, H, W, W, C, C)

    output_nodes = get_output_nodes(nodes, src_id2dst_ids,
                                    check_single_output=False)
    for node in output_nodes:
        if node.resolution is None:
            parent_node = id2node[list(dst_id2src_ids[node.str_id])[0]]
            pH = parent_node.resolution[1]
            pW = parent_node.resolution[3]
            pC = parent_node.resolution[5]
            node.resolution = (pH, pH, pW, pW, pC, pC)

    for nid in unknown_node_ids:
        if id2node[nid].resolution is None:
            resolution = derive_node_resolution(id2node[nid], id2node, src_id2dst_ids, dst_id2src_ids)
            id2node[nid].resolution = resolution
    return max_H, max_W


def get_input_nodes(nodes, src_id2dst_ids,
                    check_single_input=True):
    if len(nodes) == 1:
        return nodes[0] if check_single_input else nodes
    dst_id2src_ids = get_reverse_adj_dict(src_id2dst_ids)
    input_nodes = []
    for n in nodes:
        if n.str_id in src_id2dst_ids and \
                (n.str_id not in dst_id2src_ids or len(dst_id2src_ids[n.str_id]) == 0):
            input_nodes.append(n)
    assert len(input_nodes) > 0, "Cannot find an input node"
    if check_single_input:
        assert len(input_nodes) == 1, "Detected more than 1 input nodes"
        return input_nodes[0]
    else:
        return input_nodes


def get_output_nodes(nodes, src_id2dst_ids,
                     check_single_output=False):
    if len(nodes) == 1:
        return nodes[0] if check_single_output else nodes
    dst_id2src_ids = get_reverse_adj_dict(src_id2dst_ids)
    output_nodes = []
    for n in nodes:
        if n.str_id in dst_id2src_ids and \
                (n.str_id not in src_id2dst_ids or len(src_id2dst_ids[n.str_id]) == 0):
            output_nodes.append(n)
    assert len(output_nodes) > 0, "Cannot find an output node"
    if check_single_output:
        assert len(output_nodes) == 1, "Detected more than 1 output nodes"
        return output_nodes[0]
    else:
        return output_nodes


def prune_single_cat_add_mul_nodes(nodes, src_id2dst_ids):
    # Cat and add nodes are special, remove if there's only 1 incoming edges to them
    dst_id2src_ids = get_reverse_adj_dict(src_id2dst_ids)
    pruned_ids = set()
    for i, n in enumerate(nodes):
        if not isinstance(n, RegularNode):
            continue
        if n.label.startswith("add") or n.label.startswith("cat") or n.label.startswith("mul"):
            parent_ids = dst_id2src_ids[n.str_id]
            if len(parent_ids) > 1:
                continue
            else:
                # We prune the cat/add node
                # Change edge connection for the node by directly connecting its parents to children
                pruned_ids.add(n.str_id)
                remove_node_edges(n, src_id2dst_ids, dst_id2src_ids)
    kept_nodes = [n for n in nodes if n.str_id not in pruned_ids]
    return kept_nodes, src_id2dst_ids


def prune_internal_input_output_nodes(nodes, src_id2dst_ids):
    # Remove any internal input/output nodes
    dst_id2src_ids = get_reverse_adj_dict(src_id2dst_ids)
    pruned_ids = set()
    for i, n in enumerate(nodes):
        if isinstance(n, RegularNode) and \
            (n.label.startswith("input") or n.label.startswith("output")) and \
                n.str_id in dst_id2src_ids and n.str_id in src_id2dst_ids:
            # We prune the input/output node
            # Change edge connection for the node by directly connecting its parents to children
            pruned_ids.add(n.str_id)
            remove_node_edges(n, src_id2dst_ids, dst_id2src_ids)
    kept_nodes = [n for n in nodes if n.str_id not in pruned_ids]
    return kept_nodes, src_id2dst_ids


def prepend_input(nodes, src_id2dst_ids,
                  op2i):
    """
    If there's no input node then manually add one
    """
    input_node = get_input_nodes(nodes, src_id2dst_ids,
                                 check_single_input=True)
    if "input" != input_node.label:
        new_input_node = RegularNode(str_id="/added_input|0",
                                     label="input",
                                     op_type_idx=op2i["input"])
        H_in, _, W_in, _, C_in, _ = input_node.resolution
        new_input_node.resolution = [H_in, H_in, W_in, W_in, C_in, C_in]
        src_id2dst_ids[new_input_node.str_id].add(input_node.str_id)
        nodes.insert(0, new_input_node)
    return nodes, src_id2dst_ids


class ComputeGraph:

    def __init__(self, C_in=3, H=224, W=224, name="",
                 max_hidden_size=None, max_kernel_size=None,
                 max_derived_H=None, max_derived_W=None):
        self.name = name
        self.input_shape = (1, H, W, C_in)
        self.edge_pairs = []
        self.regular_nodes = []
        self.weighted_nodes = []
        self.n_regular_nodes = 0
        self.n_weighted_nodes = 0
        self.max_hidden_size = max_hidden_size # For normalization
        self.max_kernel_size = max_kernel_size
        self.max_derived_H = max_derived_H
        self.max_derived_W = max_derived_W

    def __deepcopy__(self, memodict={}):
        cg = ComputeGraph()
        cg.name = self.name
        cg.input_shape = self.input_shape
        cg.edge_pairs = copy.deepcopy(self.edge_pairs)
        cg.regular_nodes = [copy.deepcopy(n) for n in self.regular_nodes]
        cg.weighted_nodes = [copy.deepcopy(n) for n in self.weighted_nodes]
        cg.n_regular_nodes = self.n_regular_nodes
        cg.n_weighted_nodes = self.n_weighted_nodes
        cg.max_hidden_size = self.max_hidden_size # For normalization
        cg.max_kernel_size = self.max_kernel_size
        cg.max_derived_H = self.max_derived_H
        cg.max_derived_W = self.max_derived_W
        return cg

    @property
    def src_id2dst_ids_dict(self):
        src_id2dst_ids = collections.defaultdict(set)
        nodes = self.nodes
        for src, dst in self.edge_pairs:
            src_id2dst_ids[nodes[src].str_id].add(nodes[dst].str_id)
        return src_id2dst_ids

    @property
    def nodes(self):
        # Implicit rule of weighted node always in front, very important
        return self.weighted_nodes + self.regular_nodes

    def get_input_node(self):
        return get_input_nodes(self.nodes, self.src_id2dst_ids_dict,
                               check_single_input=True)

    def get_output_node(self):
        return get_output_nodes(self.nodes, self.src_id2dst_ids_dict,
                                check_single_output=False)

    def get_relative_node_positions(self):
        # Performs a bfs on the graph and returns a normalized position value for each op in the graph
        # The position is only a rough estimate, it does not capture the true level of each op
        nodes = self.nodes
        id2node = {n.str_id: n for n in nodes}
        src_id2dst_ids = self.src_id2dst_ids_dict
        input_node = get_input_nodes(self.nodes, src_id2dst_ids)
        visited_ids = {input_node.str_id,}
        visited_nodes = []
        q = [input_node]
        while len(q) > 0:
            node = q.pop(0)
            visited_nodes.append(node)
            neighbor_ids = sorted(src_id2dst_ids[node.str_id])
            for nid in neighbor_ids:
                if nid not in visited_ids:
                    visited_ids.add(node.str_id)
                    q.append(id2node[nid])
        id2pos = {}
        for i, n in enumerate(visited_nodes):
            id2pos[n.str_id] = float(i + 1) / len(visited_nodes)
        return [id2pos[n.str_id] for n in nodes] # Corresponds to each node in self.nodes

    def get_gnn_features(self):
        regular_node_inds = [n.op_type_idx for n in self.regular_nodes]
        regular_node_shapes = [list(n.resolution) for n in self.regular_nodes]
        for lv in regular_node_shapes:
            lv[0] = float(lv[0]) / self.max_derived_H
            lv[1] = float(lv[1]) / self.max_derived_H
            lv[2] = float(lv[2]) / self.max_derived_W
            lv[3] = float(lv[3]) / self.max_derived_W
            lv[4] = float(lv[4]) / self.max_hidden_size
            lv[5] = float(lv[5]) / self.max_hidden_size
            if not all(0 <= v <= 10 for v in lv):
                warnings.warn("Some features values are not in [0, 10]: {}".format(lv))
        weighted_node_inds = [n.op_type_idx for n in self.weighted_nodes]
        weighted_node_shapes = [list(n.resolution) for n in self.weighted_nodes]
        weighted_node_bias = [[0, 1] if n.metadata is not None and \
                                        "use_bias" in n.metadata and \
                                        n.metadata["use_bias"]
                              else [1, 0] for n in self.weighted_nodes]
        assert len(weighted_node_bias) == len(weighted_node_inds)
        for lv in weighted_node_shapes:
            lv[0] = float(lv[0]) / self.max_derived_H
            lv[1] = float(lv[1]) / self.max_derived_H
            lv[2] = float(lv[2]) / self.max_derived_W
            lv[3] = float(lv[3]) / self.max_derived_W
            lv[4] = float(lv[4]) / self.max_hidden_size
            lv[5] = float(lv[5]) / self.max_hidden_size
            if not all(0 <= v <= 10 for v in lv):
                warnings.warn("Some features values are not in [0, 10]: {}".format(lv))
        weighted_node_kernels = [n.shape[2:] for n in self.weighted_nodes]
        return regular_node_inds, regular_node_shapes, \
               weighted_node_inds, weighted_node_shapes, weighted_node_kernels, weighted_node_bias, \
               edge_pairs_to_edge_list(self.edge_pairs)

    @property
    def regular_node_start_idx(self):
        # By default, weighted nodes are group together in front
        # Regular nodes follow the weight nodes
        return self.n_weighted_nodes

    def build_from_model_maker(self, model_maker, op2idx,
                               oov_threshold=None,
                               report_time=False):
        start_time = time.time()
        g = tf.Graph()
        with g.as_default():
            sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                                                    log_device_placement=False))
            image_batch = tf.ones(self.input_shape, tf.float32)
            x = tf.identity(image_batch, "input")
            model = model_maker()
            output = model(x, training=False) ## Change here
            if type(output) is list:
                output_names = []
                for i, output_i in enumerate(output):
                    output_names.append("output_{}".format(i))
                    output_i = tf.identity(output_i, output_names[-1])
            else:
                output = tf.identity(output, "output")
                output_names = ["output"]
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            const_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_names)
        tf.reset_default_graph()
        if report_time:
            print("Model forward time: {}".format(time.time() - start_time))
        return self.build_from_graph_def(const_graph, op2idx,
                                         oov_threshold, report_time)

    def build_from_deeplab_mm(self, model_maker, op2idx,
                               oov_threshold=None,
                               report_time=False):
        start_time = time.time()
        g = tf.Graph()
        with g.as_default():
            sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                                                    log_device_placement=False))
            image_batch = tf.ones(self.input_shape, tf.float32)
            x = tf.identity(image_batch, "input")
            output = model_maker(x)
            if type(output) is list:
                output_names = []
                for i, output_i in enumerate(output):
                    output_names.append("output_{}".format(i))
                    output_i = tf.identity(output_i, output_names[-1])
            else:
                output = tf.identity(output, "output")
                output_names = ["output"]
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            const_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, output_names)
        tf.reset_default_graph()
        if report_time:
            print("Model forward time: {}".format(time.time() - start_time))
        return self.build_from_graph_def(const_graph, op2idx,
                                         oov_threshold, report_time)

    def build_from_pb(self, pb_file, op2idx,
                      oov_threshold=None,
                      report_time=False):
        # Read from pb
        with tf.gfile.GFile(pb_file, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        return self.build_from_graph_def(graph_def, op2idx,
                                         oov_threshold, report_time)

    def build_from_graph_def(self, graph_def, op2idx,
                             oov_threshold=None,
                             report_time=False):
        """
        NOTE: normalization here is applied only to the .shapes var in weighted nodes!
        self.get_gnn_features() will apply further normalizations to the regular nodes and HW values of all nodes
        It is highly unlikely that we'll normalize kernels, consider removing this in the future
        """
        # Find all nodes
        start_time = time.time()
        nodes, src_id2dst_ids = find_nodes_bfs(copy.deepcopy(graph_def), op2idx)
        if report_time:
            print("Node discovery time: {}".format(time.time() - start_time))

        # Graph simplification
        start_time = time.time()
        nodes, src_id2dst_ids = prune_nodes_by_keywords(nodes, src_id2dst_ids)
        nodes, src_id2dst_ids = merge_kernel_nodes(nodes, src_id2dst_ids)
        nodes, src_id2dst_ids = merge_bias_nodes(nodes, src_id2dst_ids)
        nodes, src_id2dst_ids = merge_bn_mean_nodes(nodes, src_id2dst_ids)
        nodes, src_id2dst_ids = re_connect_pad_nodes(nodes, src_id2dst_ids)
        nodes, src_id2dst_ids = prune_single_cat_add_mul_nodes(nodes, src_id2dst_ids)
        nodes, src_id2dst_ids = prune_internal_input_output_nodes(nodes, src_id2dst_ids)
        # self.gviz_visualize()
        max_derived_H, max_derived_W = \
            extract_node_resolutions(get_input_nodes(nodes, src_id2dst_ids), nodes, src_id2dst_ids,
                                     graph_def, self.input_shape[3], self.input_shape[1], self.input_shape[2])
        nodes, src_id2dst_ids = prepend_input(nodes, src_id2dst_ids, op2idx)

        if self.max_derived_H is None:
            self.max_derived_H = max_derived_H
        elif self.max_derived_H < max_derived_H:
            warnings.warn("Specified max_derived_H={} "
                          "but found actual max_derived_H={}".format(self.max_derived_H, max_derived_H))
        if self.max_derived_W is None:
            self.max_derived_W = max_derived_W
        elif self.max_derived_W < max_derived_W:
            warnings.warn("Specified max_derived_W={} "
                          "but found actual max_derived_W={}".format(self.max_derived_W, max_derived_W))
        if report_time:
            print("Graph simplification time: {}".format(time.time() - start_time))

        # Normalization constant checking
        start_time = time.time()
        weighted_nodes = [n for n in nodes if isinstance(n, WeightedNode)]
        hidden_size = RunningStatMeter()
        kernel_size = RunningStatMeter()
        for n in weighted_nodes:
            hidden_size.update(n.shape[0])
            hidden_size.update(n.shape[1])
            kernel_size.update(n.shape[2])
            kernel_size.update(n.shape[3])
        max_hidden_size = hidden_size.max
        max_kernel_size = kernel_size.max
        if self.max_hidden_size is None:
            self.max_hidden_size = max_hidden_size
        elif self.max_hidden_size * 10 < max_hidden_size:
            # Give some room for normalized hidden sizes >= 1.0
            warnings.warn("Specified max_hidden_size={} "
                          "but found actual max_hidden_size={}".format(self.max_hidden_size, max_hidden_size))
        if self.max_kernel_size is None:
            self.max_kernel_size = max_kernel_size
        elif self.max_kernel_size < max_kernel_size:
            warnings.warn("Specified max_kernel_size={} "
                          "but found actual max_kernel_size={}".format(self.max_kernel_size, max_kernel_size))

        if report_time:
            print("Normalization time: {}".format(time.time() - start_time))

        # Set nodes and edge pairs
        self.set_nodes_edge_pairs(nodes, src_id2dst_ids)
        if len(self.nodes) > 5000:
            warnings.warn("Detected large number of nodes, consider simplifying")
        if oov_threshold is not None:
            oov_count = 0.
            oov_labels = set()
            for node in self.nodes:
                if node.op_type_idx == op2idx.oov_idx:
                    oov_count += 1
                    oov_labels.add(node.label)
            oov_percent = oov_count / len(self)
            if oov_percent > oov_threshold:
                warnings.warn("Excessive OOV percent in graph: {}%".format(oov_percent * 100))
                print("Discovered OOV labels: {}".format(oov_labels))
        assert all(n.shape is not None for n in self.weighted_nodes), "Some weighted nodes have None shape"
        self.get_input_node()
        self.get_output_node()
        return self.nodes, src_id2dst_ids

    def set_nodes_edge_pairs(self, nodes, src_id2dst_ids):
        self.edge_pairs = []
        self.regular_nodes = [n for n in nodes if isinstance(n, RegularNode)]
        self.weighted_nodes = [n for n in nodes if isinstance(n, WeightedNode)]
        self.n_regular_nodes = len(self.regular_nodes)
        self.n_weighted_nodes = len(self.weighted_nodes)
        node_id2idx = {n.str_id: i for i, n in enumerate(self.nodes)}
        for src_id, dst_ids in src_id2dst_ids.items():
            for dst_id in list(dst_ids):
                edge = (node_id2idx[src_id], node_id2idx[dst_id])
                self.edge_pairs.append(edge)
        self.edge_pairs.sort()

    def __len__(self):
        return len(self.weighted_nodes) + len(self.regular_nodes)

    def __str__(self):
        return "ComputeGraph[{}](n_nodes: {}, n_edges: {})".format(self.name, len(self), len(self.edge_pairs))

    def __repr__(self):
        return str(self)

    def gviz_visualize(self, view=True, output_dir=None,
                       filename=None):
        if os.name == "nt":
            os.environ["PATH"] += os.pathsep + "C:\\Program Files (x86)\\Graphviz2.38\\bin\\"
        try:
            from graphviz import Digraph
        except ModuleNotFoundError:
            return
        node_attr = dict(style='filled',
                         shape='box',
                         align='left',
                         fontsize='12',
                         ranksep='0.1',
                         height='0.2')
        dot = Digraph(node_attr=node_attr, filename=filename, graph_attr=dict(size="12,12"))
        for n in self.nodes:
            if isinstance(n, WeightedNode):
                dot.node(n.str_id, str(n), fillcolor='lightblue')
            elif isinstance(n, RegularNode):
                dot.node(n.str_id, str(n))
            else:
                raise ValueError("Unknown node type: {}".format(n))
        for src_i, dst_i in self.edge_pairs:
            dot.edge(self.nodes[src_i].str_id, self.nodes[dst_i].str_id)
        _resize_graph(dot)
        dot.render(view=view, directory=output_dir,
                   format="png", cleanup=True)


def load_from_state_dict(cg:ComputeGraph, sd):
    cg.name = sd["name"]
    cg.input_shape = sd["input_shape"]
    cg.max_hidden_size = sd["max_hidden_size"]
    cg.max_kernel_size = sd["max_kernel_size"]
    cg.max_derived_H = sd["max_derived_H"]
    cg.max_derived_W = sd["max_derived_W"]
    cg.edge_pairs = sd["edge_pairs"]

    regular_nodes = []
    for n in sd["regular_nodes"]:
        node = RegularNode(str_id=n["str_id"],
                           label=n["label"],
                           op_type_idx=n["op_type_idx"])
        node.strides = n["strides"]
        node.metadata = n["metadata"]
        node.resolution = n["resolution"]
        regular_nodes.append(node)

    weighted_nodes = []
    for n in sd["weighted_nodes"]:
        node = WeightedNode(str_id=n["str_id"],
                            label=n["label"],
                            shape=n["shape"],
                            op_type_idx=n["op_type_idx"])
        node.strides = n["strides"]
        node.metadata = n["metadata"]
        node.resolution = n["resolution"]
        weighted_nodes.append(node)

    cg.regular_nodes = regular_nodes
    cg.weighted_nodes = weighted_nodes
    cg.n_regular_nodes = len(regular_nodes)
    cg.n_weighted_nodes = len(weighted_nodes)
    return cg
