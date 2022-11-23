import numpy as np
from model_src.comp_graph.tf_comp_graph import OP2I
from params import *



"""
Src: https://github.com/sovrasov/flops-counter.pytorch/blob/deaefe75eaf6f83ca713edee30e82c5da33f6863/ptflops/flops_counter.py
"""


def _count_conv2d(node, batch_size=1, num_inputs=1):
    Hin, Hout, Win, Wout, Cin, Cout = node.resolution
    k1, k2 = node.shape[2:]
    has_bias = node.metadata["use_bias"] if node.metadata is not None and "use_bias" in node.metadata else False
    output_dims = [Hout, Wout]

    kernel_dims = [k1, k2]
    in_channels = Cin
    out_channels = Cout

    filters_per_channel = out_channels
    conv_per_position_flops = int(np.prod(kernel_dims)) * \
                              in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if has_bias:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    return int(overall_flops) * num_inputs


def _count_depthwise(node, batch_size=1, num_inputs=1):
    Hin, Hout, Win, Wout, Cin, Cout = node.resolution
    k1, k2 = node.shape[2:]
    has_bias = node.metadata["use_bias"] if node.metadata is not None and "use_bias" in node.metadata else False
    output_dims = [Hout, Wout]

    kernel_dims = [k1, k2]
    in_channels = Cin
    out_channels = Cout

    filters_per_channel = out_channels // Cout
    conv_per_position_flops = int(np.prod(kernel_dims)) * \
                              in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if has_bias:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    return int(overall_flops) * num_inputs


def _count_act(node, batch_size=1, num_inputs=1):
    Hin, Hout, Win, Wout, Cin, Cout = node.resolution
    num_elements = Hout * Wout * Cout * batch_size
    return int(num_elements) * num_inputs


def _count_pool(node, batch_size=1, num_inputs=1):
    Hin, Hout, Win, Wout, Cin, Cout = node.resolution
    num_elements = Hout * Wout * Cout * batch_size
    return int(num_elements) * num_inputs


def _count_matmul(node, batch_size=1, num_inputs=1):
    Hin, Hout, Win, Wout, Cin, Cout = node.resolution
    has_bias = node.metadata["use_bias"] if node.metadata is not None and "use_bias" in node.metadata else False
    output_last_dim = Cout
    bias_flops = output_last_dim if has_bias is not None else 0
    return int(np.prod([batch_size * Hin * Win * Cin]) * output_last_dim + bias_flops) * num_inputs


def _count_bn(node, batch_size=1, num_inputs=1):
    # Assume BN is always trainable
    Hin, Hout, Win, Wout, Cin, Cout = node.resolution
    batch_flops = np.prod([batch_size * Hin * Win * Cin])
    batch_flops *= 2 # Because of affine
    return int(batch_flops) * num_inputs


def _count_add(node, batch_size=1, num_inputs=1):
    Hin, Hout, Win, Wout, Cin, Cout = node.resolution
    return int(np.prod([batch_size * Hout * Wout * Cout])) * num_inputs


def _count_mul(node, batch_size=1, num_inputs=1):
    Hin, Hout, Win, Wout, Cin, Cout = node.resolution
    return int(np.prod([batch_size * Hout * Wout * Cout])) * num_inputs


def _count_identity(node, batch_size=1, num_inputs=1):
    return 0


_COUNTER_MAP = {
    # Must match the primitive ops definitions in tf_primitive_ops_simplified.txt
    "conv": _count_conv2d,
    "depthwise": _count_depthwise,
    "relu": _count_act,
    "relu6": _count_act,
    "matmul": _count_matmul,
    "add": _count_add,
    "mul": _count_mul,
    "avgpool": _count_pool,
    "maxpool": _count_pool,
    "mean": _count_pool,
    "global": _count_pool,
    "sigmoid": _count_act,
    "tanh": _count_act,
    "swish": _count_act,
    "batchnorm": _count_bn,
    "identity": _count_identity,
    "zero": _count_identity,
    "input": _count_identity,
    "output": _count_identity,
    "concat": _count_identity,
    "paddings": _count_identity,
}


def get_flops(op2i:OP2I, nodes, input_inds,
              custom_op_counters=None,
              batch_size=1, divisor=1e6,
              allow_zero_flops=False,
              log_f=print):
    total_flops = 0.
    seen_ops = set()
    for ni, node in enumerate(nodes):
        assert hasattr(node, "op_type_idx"), "Invalid node type: {}".format(str(node))
        node_input_inds = input_inds[ni]
        node_op = op2i.query_op(node.op_type_idx).lower()

        if custom_op_counters is not None and \
            node_op in custom_op_counters:
            counter = custom_op_counters[node_op]
            if node_op not in seen_ops:
                log_f("Using custom counter for op: {}".format(node_op))
                seen_ops.add(node)

        elif node_op in _COUNTER_MAP:
            counter = _COUNTER_MAP[node_op]
            if node_op not in seen_ops:
                log_f("Using native counter for op: {}".format(node_op))
                seen_ops.add(node)

        else:
            if node_op not in seen_ops:
                log_f("Undefined op: {}. Treat as zero flops".format(node_op))
                seen_ops.add(node)
            continue

        op_flops = counter(node,
                           batch_size=batch_size,
                           num_inputs=max(len(node_input_inds), 1))
        total_flops += op_flops

    total_flops /= divisor

    if allow_zero_flops and total_flops < 1e-7:
        return None
    else:
        assert total_flops > 0., \
            "Zero FLOPs encountered on nodes: {}, input_inds: {}".format(nodes, input_inds)

    return total_flops

