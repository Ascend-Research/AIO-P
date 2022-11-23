import math
import copy
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn import functional as F
from utils.graph_utils import topo_sort_dfs, get_reverse_adj_dict
from model_src.comp_graph.tf_comp_graph import OP2I, ComputeGraph, WeightedNode, remove_node_edges


BN_MOMENTUM = 0.1
BN_EPSILON = 1e-3
L2_REG_CONSTANT = 1e-5
torch.autograd.set_detect_anomaly(True)


class Identity(nn.Identity):

    def __init__(self, name='Identity'):
        self.name = name
        super(Identity, self).__init__()

    def forward(self, x, *args):
        return super(Identity, self).forward(x)


class Zero(nn.Module):

    def __init__(self, stride=1, name='Zero'):
        self.name = name
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class Input(nn.Module):
    """
    Just a dummy node
    """

    def __init__(self, name="Input"):
        self._name = name
        super(Input, self).__init__()

    def forward(self, x, *args):
        return x


class Output(nn.Module):
    """
    Just a dummy node
    """

    def __init__(self, name="Output"):
        self._name = name
        super(Output, self).__init__()

    def forward(self, x, *args):
        return x


class Add(nn.Module):
    """
    Just a dummy node, actual op is performed in network
    """

    def __init__(self, name="Add"):
        self._name = name
        super(Add, self).__init__()

    def forward(self, x, *args):
        return x


class Mul(nn.Module):
    """
    Just a dummy node, actual op is performed in network
    """

    def __init__(self, name="Mul"):
        self._name = name
        super(Mul, self).__init__()

    def forward(self, x, *args):
        return x


class Concat(nn.Module):
    """
    Just a dummy node, actual op is performed in network
    """

    def __init__(self, name="Concat"):
        self._name = name
        super(Concat, self).__init__()

    def forward(self, x, *args):
        return x


class Activation(nn.Module):
    def __init__(self, act_func, name="Activation"):
        self.act_func = act_func
        self._name = name
        super(Activation, self).__init__()

    def forward(self, x, *args):
        return self.act_func().forward(x)


class Conv2d(nn.Conv2d):
    """2D Convolutions like TensorFlow, for a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """

    # tf:  kernel_regularizer=regularizers.l2(L2_REG_CONSTANT), kernel_initializer="he_uniform"
    def __init__(self, C_in, C_out, kernel_size, stride=1, padding="same", bias=True, dilation=1, groups=1,
                 name="Conv2d"):
        self._name = name
        self.padding_type = padding

        if 0 in kernel_size:
            kernel_size = [1, 1]
        super(Conv2d, self).__init__(C_in, C_out, kernel_size, stride=stride, bias=bias, dilation=dilation,
                                     groups=groups)

    def forward(self, x, *args):
        if self.padding_type.casefold() == "same":
            x = same_pad(self, x)
        return super(Conv2d, self).forward(x)


class ConvTranspose2d(nn.ConvTranspose2d):
    """2D Deconvolution like Tensorflow, for a dynamic image size.
    """

    def __init__(self, C_in, C_out, kernel_size, stride=1, bias=True, dilation=1, groups=1, name="ConvTranspose2d"):
        self._name = name
        from model_src.multitask.task_adapter import get_deconv_cfg
        padding, output_padding = get_deconv_cfg(kernel_size[0])
        super(ConvTranspose2d, self).__init__(C_in, C_out, kernel_size, stride=stride, bias=bias, dilation=dilation,
                                              groups=groups, padding=padding, output_padding=output_padding)

    def forward(self, x, *args):
        return super(ConvTranspose2d, self).forward(x)


class DepthwiseConv2d(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding="same", affine=True, bias=False, dilation=1,
                 name="DepthwiseConv2d"):
        # tf:  kernel_regularizer=regularizers.l2(L2_REG_CONSTANT), kernel_initializer="he_uniform"
        assert C_in == C_out, "DepthwiseConv2d does not support channel size change for now"
        self._name = name
        self.padding_type = padding
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size
        super(DepthwiseConv2d, self).__init__()
        self.op = nn.Sequential(
            Conv2d(C_in, C_in, kernel_size, stride=stride, padding=padding,
                   groups=C_in, bias=bias, dilation=dilation, name=self._name + "Conv2d"),
        )

    def forward(self, x):
        return self.op(x)


class BatchNorm2d(nn.BatchNorm2d):  # how to include axis
    def __init__(self, C_in, C_out, epsilon=BN_EPSILON, momentum=BN_MOMENTUM,
                 name="BatchNorm2d", affine=True):
        self._name = name
        # C_in == C_out
        # affine = affine if affine == True else trainable
        super(BatchNorm2d, self).__init__(C_in, eps=epsilon, momentum=momentum, affine=affine)

    def forward(self, x, affine=None, *args):
        if affine is not None:
            self.affine = affine
        return super(BatchNorm2d, self).forward(x)


class Linear(nn.Linear):  # Cin = Cout?
    def __init__(self, C_in, C_out, name="Linear"):
        self._name = name
        super(Linear, self).__init__(C_in, C_out, bias=True)

    def forward(self, x, *args):
        x = torch.reshape(x, (-1, 1, 1, list(x.shape)[1]))
        out = super(Linear, self).forward(x)
        return torch.reshape(out, (-1, list(out.shape)[-1], 1, 1))


class MaxPool2d(nn.MaxPool2d):  #
    def __init__(self, kernel_size, stride=None, padding="same", name="MaxPool2d"):
        self._name = name
        self.padding_type = padding
        super(MaxPool2d, self).__init__(kernel_size, stride=stride, padding=0)

    def forward(self, x, *args):
        if self.padding_type.casefold() == "same":
            x = same_pad(self, x)
        return super(MaxPool2d, self).forward(x)


class AvgPool2d(nn.AvgPool2d):  #
    def __init__(self, kernel_size, stride=None, padding="same", name="AvgPool2d"):
        self._name = name
        self.padding_type = padding
        super(AvgPool2d, self).__init__(kernel_size, stride=stride, padding=0)

    def forward(self, x, *args):
        if self.padding_type.casefold() == "same":
            x = same_pad(self, x)
        return super(AvgPool2d, self).forward(x)


class GlobalAveragePooling2d(nn.AdaptiveAvgPool2d):  #
    def __init__(self, name="GlobalAveragePooling2d"):
        self._name = name
        super(GlobalAveragePooling2d, self).__init__((1, 1))

    def forward(self, x, *args):
        return super(GlobalAveragePooling2d, self).forward(x)


class ZeroPad2d(nn.ZeroPad2d):  #
    def __init__(self, padding=0, name="ZeroPad2d"):
        self._name = name
        super(ZeroPad2d, self).__init__(padding=padding)

    def forward(self, x, *args):
        return super(ZeroPad2d, self).forward(x)


def same_pad(module: nn.Module, x):
    # Tips for 'SAME' mode padding.
    #     Given the following:
    #         i: width or height
    #         s: stride
    #         k: kernel size
    #         d: dilation
    #         p: padding
    #     Output after Conv2d:
    #         o = floor((i+p-((k-1)*d+1))/s+1)
    # If o equals i, i = floor((i+p-((k-1)*d+1))/s+1),
    # => p = (i-1)*s+((k-1)*d+1)-i
    ih, iw = x.size()[-2:]
    sh, sw = module.stride
    if isinstance(module, Conv2d):
        kh, kw = module.weight.size()[-2:]
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride ! ! !
        pad_h = max((oh - 1) * sh + (kh - 1) * module.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * sw + (kw - 1) * module.dilation[1] + 1 - iw, 0)
    else:
        kh, kw = module.kernel_size[0], module.kernel_size[1]
        d = 1 if isinstance(module, AvgPool2d) else module.dilation
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride ! ! !
        pad_h = max((oh - 1) * sh + (kh - 1) * d + 1 - ih, 0)
        pad_w = max((ow - 1) * sw + (kw - 1) * d + 1 - iw, 0)
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    return x


def get_op_model_for_node(node, op2i: OP2I, scope_name,
                          op_override_dict=None):
    # Should check tf_primitive_ops_simplified.txt to make sure all is covered
    op_type = op2i.query_op(node.op_type_idx)

    if op_override_dict is not None and \
        op_type in op_override_dict:
        op_type = op_override_dict[op_type]

    if op_type == "conv":
        stride = node.strides[1:-1] if node.strides is not None else (1, 1)
        use_bias = node.metadata["use_bias"] if node.metadata is not None and "use_bias" in node.metadata else False
        padding = node.metadata["padding"] if node.metadata is not None and "padding" in node.metadata else "same"
        dil_rate = node.metadata["dil_rate"] if node.metadata is not None and "dil_rate" in node.metadata else 1
        if "_transpose" in node.label:
            op = ConvTranspose2d(node.resolution[-2], node.resolution[-1], node.shape[2:], stride=stride,
                                 dilation=dil_rate, bias=use_bias, name=scope_name)
        else:
            op = Conv2d(node.resolution[-2], node.resolution[-1], node.shape[2:], stride=stride,
                        dilation=dil_rate, padding=padding, bias=use_bias, name=scope_name)
    elif op_type == "depthwise":
        stride = node.strides[1:-1] if node.strides is not None else (1, 1)
        use_bias = node.metadata["use_bias"] if node.metadata is not None and "use_bias" in node.metadata else False
        padding = node.metadata["padding"] if node.metadata is not None and "padding" in node.metadata else "same"
        dil_rate = node.metadata["dil_rate"] if node.metadata is not None and "dil_rate" in node.metadata else 1
        op = DepthwiseConv2d(node.resolution[-2], node.resolution[-1], kernel_size=node.shape[2:], stride=stride,
                             dilation=dil_rate, padding=padding, name=scope_name, bias=use_bias)
    elif op_type == "batchnorm":
        op = BatchNorm2d(node.resolution[-2], node.resolution[-1], affine=True, name=scope_name)
    elif op_type == "relu":
        op = Activation(nn.ReLU, name=scope_name)
    elif op_type == "relu6":
        op = Activation(nn.ReLU6, name=scope_name)
    elif op_type == "matmul":
        op = Linear(node.resolution[-2], node.resolution[-1], name=scope_name)
    elif op_type == "add":
        op = Add(name=scope_name)
    elif op_type == "mul":
        op = Mul(name=scope_name)
    elif op_type == "input":
        op = Input(name=scope_name)
    elif op_type == "output":
        op = Output(name=scope_name)
    elif op_type == "maxpool":
        stride = node.strides[1:-1] if node.strides is not None else (1, 1)
        padding = node.metadata["padding"] if node.metadata is not None and "padding" in node.metadata else "same"
        op = MaxPool2d(node.metadata["ksize"][1:-1], stride=stride, padding=padding,
                       name=scope_name)
    elif op_type == "avgpool":
        stride = node.strides[1:-1] if node.strides is not None else (1, 1)
        padding = node.metadata["padding"] if node.metadata is not None and "padding" in node.metadata else "same"
        op = AvgPool2d(node.metadata["ksize"][1:-1], stride=stride, padding=padding,
                       name=scope_name)
    elif op_type == "paddings":
        h1, h2 = node.metadata["pad_h"]
        w1, w2 = node.metadata["pad_w"]
        op = ZeroPad2d((w1, w2, h1, h2), name=scope_name)
    elif op_type == "sigmoid":
        op = Activation(nn.Sigmoid, name=scope_name)
    elif op_type == "tanh":
        op = Activation(nn.Tanh, name=scope_name)
    elif op_type == "gelu":
        op = Activation(nn.GELU, name=scope_name)
    elif op_type == "mean" or op_type == "global":
        op = GlobalAveragePooling2d(name=scope_name)
    elif op_type == "identity":
        if node.metadata is not None and 'align_corners' in node.metadata.keys():
            op = DeepLabBilinear((int(node.resolution[1]), int(node.resolution[3])),
                                 node.metadata['align_corners'])
        else:
            op = Identity(name=scope_name)
    elif op_type == "zero":
        op = Zero(name=scope_name)
    elif op_type == "concat":
        op = Concat(name=scope_name)
    else:
        raise ValueError("Unknown op_type: {}, cannot covert to output network".format(op_type))
    return op_type, op


class DeepLabBilinear(nn.Module):
    def __init__(self, size, align):
        super(DeepLabBilinear, self).__init__()
        self.size = size
        self.align = align

    def forward(self, x):
        if self.size[0] > 0:
            return nn.functional.interpolate(x, size=self.size, mode='bilinear', align_corners=self.align)
        else:
            return x


def get_output_net_op_graph(cg: ComputeGraph):
    src2dst_ids = cg.src_id2dst_ids_dict
    dst2src_ids = get_reverse_adj_dict(src2dst_ids)
    id2node = {n.str_id: n for n in cg.nodes}
    node_ids = id2node.keys()
    sorted_ids = topo_sort_dfs(node_ids, dst2src_ids)
    assert len(sorted_ids) == len(node_ids)
    topo_nodes = [id2node[_id] for _id in sorted_ids]
    net_input_inds = []
    node_id2idx = {n.str_id: i for i, n in enumerate(topo_nodes)}
    for ni, n in enumerate(topo_nodes):
        input_ids = dst2src_ids[n.str_id]
        input_inds = [node_id2idx[_id] for _id in input_ids]
        input_inds.sort()
        assert all(i < ni for i in input_inds)
        net_input_inds.append(input_inds)
    return topo_nodes, net_input_inds


def handle_tensor_channel_mismatch(tensors,
                                   desired_resolution):  # , data_format="channels_last"
    channel_dim = 1
    if len(tensors) <= 1:
        # Single input case, must check against desired resolutions
        new_tensors = []
        for t in tensors:
            shape = list(t.shape)
            if shape[channel_dim] > desired_resolution[-2]:
                t = t[:, :desired_resolution[-2], :, :]
            elif shape[channel_dim] < desired_resolution[-2]:
                raise NotImplementedError(
                    "Cannot handle smaller than desired input channel size: {} vs. {}".
                        format(shape[channel_dim], desired_resolution[-2]))
            new_tensors.append(t)
        return new_tensors

    base_shape = list(tensors[0].shape)
    min_shape_vals = [v for v in base_shape]
    mismatch_found = False
    for t in tensors:
        shape = list(t.shape)
        for si, v in enumerate(shape):
            if v is None:
                continue
            elif si == 0 and v != base_shape[si]:
                raise ValueError("Cannot handle batch dim mismatches: {}, {}".format(base_shape, shape))
            elif si == channel_dim and v != base_shape[si]:
                mismatch_found = True
            min_shape_vals[si] = min(min_shape_vals[si], v)
    if not mismatch_found:
        return tensors
    new_tensors = []
    for t in tensors:
        shape = list(t.shape)
        if shape[channel_dim] != min_shape_vals[channel_dim]:
            t = t[:, :min_shape_vals[channel_dim], :, :]
        new_tensors.append(t)
    return new_tensors


def post_prune_dilation(cg: ComputeGraph,
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


class CompGraphOutputNet(nn.Module):

    def __init__(self, op2i=None, cg=None,
                 topo_nodes=None,
                 net_input_inds=None,
                 squeeze_output=True,
                 l2_reg_constant=L2_REG_CONSTANT,
                 name="CompGraphOutputNet",
                 data_format="channels_last",
                 op_override_dict=None):
        super(CompGraphOutputNet, self).__init__()
        # assert data_format == "channels_last"# "Currently only support channels_last"
        self._name = name
        # self.data_format = data_format
        self.squeeze_output = squeeze_output
        # self.ops = []

        self.ops_dict = OrderedDict()
        if cg is not None:
            topo_nodes, net_input_inds = get_output_net_op_graph(cg)
        else:
            assert topo_nodes is not None and net_input_inds is not None, \
                "Either provide a compute graph object or the graph itself"
        if op2i is None:
            op2i = OP2I().build_from_file()
        self.output_node_inds = self.get_output_node_inds(topo_nodes, net_input_inds)
        self.net_input_inds = net_input_inds
        self.bn_inds = set()
        self.mul_inds = set()
        self.add_inds = set()
        self.mean_inds = set()
        self.concat_inds = set()
        self.opi2io_shapes = {}
        for ni, node in enumerate(topo_nodes):
            self.opi2io_shapes[ni] = node.resolution
            scope_name = "op_{}/{}".format(ni, op2i.query_op(node.op_type_idx))
            op_type, op_model = get_op_model_for_node(node, op2i, scope_name,
                                                      op_override_dict=op_override_dict)
            self.ops_dict[scope_name] = op_model
            if op_type == "batchnorm":
                self.bn_inds.add(ni)
            elif op_type == "mul":
                self.mul_inds.add(ni)
            elif op_type == "add":
                self.add_inds.add(ni)
            elif op_type == "mean":
                self.mean_inds.add(ni)
            elif op_type == "concat":
                self.concat_inds.add(ni)
        self.ops = nn.ModuleDict(self.ops_dict)

    @staticmethod
    def get_output_node_inds(topo_nodes, net_input_inds):
        non_output_node_inds = set()
        for input_inds in net_input_inds:
            for i in input_inds:
                non_output_node_inds.add(i)
        output_node_inds = set()
        for ni, node in enumerate(topo_nodes):
            if ni not in non_output_node_inds:
                output_node_inds.add(ni)
        return output_node_inds

    def get_name(self):
        return self._name

    def forward(self, x):
        op_outputs = []
        for i, op_name in enumerate(self.ops.keys()):
            input_inds = self.net_input_inds[i]
            if len(input_inds) == 0:
                # It's an input node
                if i in self.bn_inds:
                    op_output = self.ops[op_name].forward(x)
                else:
                    op_output = self.ops[op_name].forward(x)
            else:
                op_inputs = [op_outputs[j] for j in input_inds]
                if i in self.concat_inds:
                    op_output = torch.cat(op_inputs, dim=1)
                    op_output = self.ops[op_name].forward(op_output)
                else:
                    op_inputs = handle_tensor_channel_mismatch(op_inputs, self.opi2io_shapes[i])
                    if i in self.bn_inds:
                        op_input = sum(op_inputs) if len(op_inputs) > 1 else op_inputs[0]
                        op_output = self.ops[op_name].forward(op_input)
                    elif i in self.add_inds:
                        # Since tensor slicing ops will not be captured by the CG
                        # Here we'll handle potential shape mismatches naively
                        op_input = sum(op_inputs) if len(op_inputs) > 1 else op_inputs[0]
                        op_output = self.ops[op_name].forward(op_input)
                    elif i in self.mul_inds:
                        op_input = op_inputs[0]
                        for n in op_inputs[1:]:
                            op_input = op_input * n
                        op_output = self.ops[op_name].forward(op_input)
                    elif i in self.mean_inds:
                        op_input = sum(op_inputs) if len(op_inputs) > 1 else op_inputs[0]
                        op_output = self.ops[op_name].forward(op_input)
                        op_output = torch.reshape(op_output, (-1, list(op_output.shape)[1], 1, 1))
                    else:
                        op_input = sum(op_inputs) if len(op_inputs) > 1 else op_inputs[0]
                        op_output = self.ops[op_name].forward(op_input)
            op_outputs.append(op_output)
        if len(self.output_node_inds) < 2:
            final_out = op_outputs[-1]
            if self.squeeze_output:
                final_out = torch.reshape(final_out, (-1, list(final_out.shape)[1]))
        else:
            final_out = []
            for oi in list(self.output_node_inds):
                out = op_outputs[oi]
                final_out.append(out)
        return final_out

