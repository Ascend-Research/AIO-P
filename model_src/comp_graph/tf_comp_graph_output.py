import tensorflow.compat.v1 as tf
from tensorflow.keras import regularizers
from utils.graph_utils import topo_sort_dfs, get_reverse_adj_dict
from model_src.comp_graph.tf_comp_graph import OP2I, ComputeGraph


BN_MOMENTUM = 0.99
BN_EPSILON = 1e-3
L2_REG_CONSTANT = 1e-5


class Identity(tf.keras.layers.Layer):

    def __init__(self, name="Identity"):
        super(Identity, self).__init__()
        self._name = name

    def call(self, inputs):
        return tf.identity(inputs, self._name)


class Zero(tf.keras.layers.Layer):

    def __init__(self, name="Zero"):
        super(Zero, self).__init__()
        self._name = name

    def call(self, inputs):
        with tf.name_scope(self._name):
            return tf.identity(inputs, self._name) * 0.


class Input(tf.keras.layers.Layer):
    """
    Just a dummy node
    """
    def __init__(self, name="Input"):
        super(Input, self).__init__()
        self._name = name

    def call(self, inputs):
        return inputs


class Output(tf.keras.layers.Layer):
    """
    Just a dummy node
    """
    def __init__(self, name="Output"):
        super(Output, self).__init__()
        self._name = name

    def call(self, inputs):
        return inputs


class Add(tf.keras.layers.Layer):
    """
    Just a dummy node, actual op is performed in network
    """
    def __init__(self, name="Add"):
        super(Add, self).__init__()
        self._name = name

    def call(self, inputs):
        return inputs


class Mul(tf.keras.layers.Layer):
    """
    Just a dummy node, actual op is performed in network
    """
    def __init__(self, name="Mul"):
        super(Mul, self).__init__()
        self._name = name

    def call(self, inputs):
        return inputs


class Concat(tf.keras.layers.Layer):
    """
    Just a dummy node, actual op is performed in network
    """
    def __init__(self, name="Concat"):
        super(Concat, self).__init__()
        self._name = name

    def call(self, inputs):
        return inputs


class Activation(tf.keras.layers.Layer):

    def __init__(self, act_func, name="Activation"):
        super(Activation, self).__init__()
        self.act_func = act_func
        self._name = name

    def call(self, inputs):
        with tf.name_scope(self._name):
            return self.act_func(inputs)


def get_op_model_for_node(node, op2i:OP2I, scope_name, l2_reg_constant,
                          data_format="channels_last"):
    op_type = op2i.query_op(node.op_type_idx)
    if op_type == "conv":
        padding = node.metadata["padding"] if node.metadata is not None and "padding" in node.metadata else "SAME"
        strides = node.strides[1:-1] if node.strides is not None else (1, 1)
        use_bias = node.metadata["use_bias"] if node.metadata is not None and "use_bias" in node.metadata else False
        dil_rate = node.metadata["dil_rate"] if node.metadata is not None and "dil_rate" in node.metadata else 1
        if "_transpose" in node.label:
            op = tf.keras.layers.Conv2DTranspose(node.resolution[-1], node.shape[2:],
                                                 strides=strides, dilation_rate=dil_rate,
                                                 padding=padding, data_format=data_format,
                                                 activation=None, use_bias=use_bias,
                                                 kernel_regularizer=regularizers.l2(l2_reg_constant),
                                                 kernel_initializer="he_uniform", name=scope_name)
        else:
            op = tf.keras.layers.Conv2D(node.resolution[-1], node.shape[2:],
                                        strides=strides, dilation_rate=dil_rate,
                                        padding=padding, data_format=data_format,
                                        activation=None, use_bias=use_bias,
                                        kernel_regularizer=regularizers.l2(l2_reg_constant),
                                        kernel_initializer="he_uniform", name=scope_name)
    elif op_type == "depthwise":
        padding = node.metadata["padding"] if node.metadata is not None and "padding" in node.metadata else "SAME"
        strides = node.strides[1:-1] if node.strides is not None else (1, 1)
        use_bias = node.metadata["use_bias"] if node.metadata is not None and "use_bias" in node.metadata else False
        dil_rate = node.metadata["dil_rate"] if node.metadata is not None and "dil_rate" in node.metadata else 1
        op = tf.keras.layers.DepthwiseConv2D(kernel_size=node.shape[2:], strides=strides,
                                             dilation_rate=dil_rate, padding=padding,
                                             data_format=data_format, name=scope_name, use_bias=use_bias,
                                             kernel_regularizer=regularizers.l2(l2_reg_constant),
                                             kernel_initializer="he_uniform")
    elif op_type == "batchnorm":
        if data_format == "channels_last":
            channel_axis = 3
        else:
            channel_axis = 1
        op = tf.keras.layers.BatchNormalization(axis=channel_axis, trainable=True,
                                                momentum=BN_MOMENTUM, epsilon=BN_EPSILON,
                                                name=scope_name)
    elif op_type == "relu":
        op = tf.keras.layers.ReLU(name=scope_name)
    elif op_type == "relu6":
        op = tf.keras.layers.ReLU(name=scope_name, max_value=6)
    elif op_type == "matmul":
        op = tf.keras.layers.Dense(node.resolution[-1], name=scope_name)
    elif op_type == "add":
        op = Add(name=scope_name)
    elif op_type == "mul":
        op = Mul(name=scope_name)
    elif op_type == "input":
        op = Input(name=scope_name)
    elif op_type == "output":
        op = Output(name=scope_name)
    elif op_type == "maxpool":
        strides = node.strides[1:-1] if node.strides is not None else (1, 1)
        padding = node.metadata["padding"] if node.metadata is not None and "padding" in node.metadata else "SAME"
        op = tf.keras.layers.MaxPooling2D(node.metadata["ksize"][1:-1], strides=strides, padding=padding,
                                          data_format=data_format, name=scope_name)
    elif op_type == "avgpool":
        strides = node.strides[1:-1] if node.strides is not None else (1, 1)
        padding = node.metadata["padding"] if node.metadata is not None and "padding" in node.metadata else "SAME"
        op = tf.keras.layers.AveragePooling2D(node.metadata["ksize"][1:-1], strides=strides, padding=padding,
                                              data_format=data_format, name=scope_name)
    elif op_type == "paddings":
        h1, h2 = node.metadata["pad_h"]
        w1, w2 = node.metadata["pad_w"]
        op = tf.keras.layers.ZeroPadding2D(((h1, h2), (w1, w2)), data_format=data_format, name=scope_name)
    elif op_type == "sigmoid":
        op = Activation(tf.keras.activations.sigmoid, name=scope_name)
    elif op_type == "tanh":
        op = Activation(tf.keras.activations.tanh, name=scope_name)
    elif op_type == "gleu":
        op = Activation(tf.keras.activations.gelu, name=scope_name)
    elif op_type == "mean" or op_type == "global":
        op = tf.keras.layers.GlobalAveragePooling2D(data_format, name=scope_name)
    elif op_type == "identity":
        if node.metadata is not None and 'align_corners' in node.metadata.keys():
            op = DeepLabBilinear((int(node.resolution[1]), int(node.resolution[3])),
                                 node.metadata['align_corners'], scope_name=scope_name)
        else:
            op = Identity(name=scope_name)
    elif op_type == "zero":
        op = Zero(name=scope_name)
    elif op_type == "concat":
        op = Concat(name=scope_name)
    else:
        raise ValueError("Unknown op_type: {}, cannot covert to output network".format(op_type))
    return op_type, op


class DeepLabBilinear(tf.keras.layers.Layer):
    def __init__(self, size, align, scope_name):
        super(DeepLabBilinear, self).__init__()
        self.size = size
        self.align = align
        self.scope_name = scope_name
    
    def call(self, inputs, training=True):
        if self.size[0] > 0:
            return tf.image.resize_bilinear(inputs, self.size, align_corners=self.align, name=self.scope_name)
        else:
            return inputs


def get_output_net_op_graph(cg:ComputeGraph):
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
                                   desired_resolution,
                                   data_format="channels_last"):
    if data_format == "channels_last":
        channel_dim = 3
    else:
        channel_dim = 1

    if len(tensors) <= 1:
        # Single input case, must check against desired resolutions
        new_tensors = []
        for t in tensors:
            shape = t.get_shape().as_list()
            if shape[channel_dim] > desired_resolution[-2]:
                t = t[:, :, :, :desired_resolution[-2]]
            elif shape[channel_dim] < desired_resolution[-2]:
                raise NotImplementedError(
                    "Cannot handle smaller than desired input channel size: {} vs. {}".
                        format(shape[channel_dim], desired_resolution[-2]))
            new_tensors.append(t)
        return new_tensors
    else:
        # Multi-input case, slice according to the min channel size
        base_shape = tensors[0].get_shape().as_list()
        min_shape_vals = [v for v in base_shape]
        mismatch_found = False
        for t in tensors:
            shape = t.get_shape().as_list()
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
            shape = t.get_shape().as_list()
            if shape[channel_dim] != min_shape_vals[channel_dim]:
                t = t[:, :, :, :min_shape_vals[channel_dim]]
            new_tensors.append(t)
        return new_tensors


class CompGraphOutputNet(tf.keras.Model):

    def __init__(self, op2i, cg=None,
                 topo_nodes=None,
                 net_input_inds=None,
                 squeeze_output=True,
                 l2_reg_constant=L2_REG_CONSTANT,
                 name="CompGraphOutputNet",
                 data_format="channels_last"):
        super(CompGraphOutputNet, self).__init__()
        assert data_format == "channels_last", "Currently only support channels_last"
        self._name = name
        self.data_format = data_format
        self.squeeze_output = squeeze_output
        self.ops = []
        if cg is not None:
            topo_nodes, net_input_inds = get_output_net_op_graph(cg)
        else:
            assert topo_nodes is not None and net_input_inds is not None, \
                "Either provide a compute graph object or the graph itself"
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
                                                      l2_reg_constant=l2_reg_constant,
                                                      data_format=self.data_format)
            self.ops.append(op_model)
            if op_type == "batchnorm":
                self.bn_inds.add(ni)
            elif op_type == "mul":
                self.mul_inds.add(ni)
            elif op_type == "add":
                self.add_inds.add(ni)
            elif op_type == "mean" or op_type == "global":
                self.mean_inds.add(ni)
            elif op_type == "concat":
                self.concat_inds.add(ni)

    def get_name(self):
        return self._name

    def call(self, inputs, training=True):
        with tf.name_scope(self._name):
            op_outputs = []
            for i, op in enumerate(self.ops):
                input_inds = self.net_input_inds[i]
                if len(input_inds) == 0:
                    # It's an input node
                    if i in self.bn_inds:
                        op_output = op(inputs, training=training)
                    else:
                        op_output = op(inputs)
                else:
                    op_inputs = [op_outputs[j] for j in input_inds]
                    if i in self.concat_inds:
                        op_output = tf.concat(op_inputs, axis=-1)
                        op_output = op(op_output)
                    else:
                        op_inputs = handle_tensor_channel_mismatch(op_inputs, self.opi2io_shapes[i])
                        if i in self.bn_inds:
                            op_input = tf.add_n(op_inputs) if len(op_inputs) > 1 else op_inputs[0]
                            op_output = op(op_input, training=training)
                        elif i in self.add_inds:
                            # Since tensor slicing ops will not be captured by the CG
                            # Here we'll handle potential shape mismatches naively
                            try:
                                op_input = tf.add_n(op_inputs) if len(op_inputs) > 1 else op_inputs[0]
                                op_output = op(op_input)
                            except:
                                print("Here here")
                                print(op)
                                print(i)
                        elif i in self.mul_inds:
                            op_input = op_inputs[0]
                            for n in op_inputs[1:]:
                                op_input = op_input * n
                            op_output = op(op_input)
                        elif i in self.mean_inds:
                            op_input = tf.add_n(op_inputs) if len(op_inputs) > 1 else op_inputs[0]
                            op_output = op(op_input)
                            op_output = tf.reshape(op_output, (-1, 1, 1,
                                                               op_output.get_shape().as_list()[-1]))
                        else:
                            op_input = tf.add_n(op_inputs) if len(op_inputs) > 1 else op_inputs[0]
                            op_output = op(op_input)

                op_outputs.append(op_output)
            final_out = op_outputs[-1]
            if self.squeeze_output:
                final_out = tf.reshape(final_out, (-1, final_out.get_shape().as_list()[-1]))
            return final_out

