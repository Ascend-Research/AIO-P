import numpy as np
import tensorflow.compat.v1 as tf
from utils.math_utils import make_divisible
from model_src.search_space.ofa_profile.arch_utils import get_final_channel_size


BN_MOMENTUM = 0.9
BN_EPSILON = 1e-3


def _decode_mbconv_e_k(op):
    _op = op.replace("mbconv", "")
    args = _op.split("_")[1:]
    e, k = args
    e = int(e.replace("e", ""))
    k = int(k.replace("k", ""))
    return e, k


def get_tf_op(op_name, C_in, C_out, stride, scope_name,
              data_format="channels_last", act_type="relu", se_ratio=0.):
    if op_name.startswith("mbconv2"):
        e, k = _decode_mbconv_e_k(op_name)
        return MBConv2(C_in, C_out, k, stride, name=scope_name, data_format=data_format,
                       expansion_ratio=e, enable_skip=stride==1 and C_in==C_out)
    elif op_name.startswith("mbconv3"):
        e, k = _decode_mbconv_e_k(op_name)
        return MBConv3(C_in, C_out, k, stride, name=scope_name, data_format=data_format,
                       expansion_ratio=e, se_ratio=se_ratio, enable_skip=stride==1 and C_in==C_out,
                       act_type=act_type)
    else:
        raise ValueError("Unknown op name: {}".format(op_name))


class AugmentedConv2D(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, stride, padding, activation, name,
                 data_format="channels_last", use_bn=True, use_bias=False):
        super(AugmentedConv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_bn = use_bn
        self.padding = padding
        self.data_format = data_format
        self.activation = activation
        self._name = name
        self.use_bias = use_bias
        if self.data_format == "channels_last":
            self.channel_axis = 3
        else:
            self.channel_axis = 1
        self.conv = tf.keras.layers.Conv2D(self.filters, self.kernel_size, strides=self.stride,
                                           padding=self.padding, data_format=self.data_format,
                                           activation=None, use_bias=self.use_bias,
                                           kernel_initializer="he_normal", name=self._name)
        if self.use_bn:
            self.bn = tf.keras.layers.BatchNormalization(axis=self.channel_axis, trainable=True,
                                                         momentum=BN_MOMENTUM, epsilon=BN_EPSILON,
                                                         name=self._name + "/bn")
        else:
            self.bn = None

    def call(self, inputs, training=True):
        with tf.name_scope(self._name):
            x = self.conv(inputs)
            if self.bn is not None:
                x = self.bn(x, training=training)
            if self.activation is not None:
                x = self.activation.call(x)
        return x


class HSigmoid(tf.keras.layers.Layer):

    def __init__(self, name="HSigmoid"):
        super(HSigmoid, self).__init__()
        self._name = name

    def call(self, x, training=True):
        with tf.name_scope(self._name):
            return tf.nn.relu6(x + 3.) / 6.


class HSwish(tf.keras.layers.Layer):

    def __init__(self, name="HSwish"):
        super(HSwish, self).__init__()
        self._name = name

    def call(self, inputs, **kwargs):
        with tf.name_scope(self._name):
            return inputs * tf.nn.relu6(inputs + np.float32(3)) * (1. / 6.)


class SqueezeExcite(tf.keras.layers.Layer):

    def __init__(self, C_in, C_squeeze, name="SqueezeExcite", data_format="channels_last"):
        super(SqueezeExcite, self).__init__()
        self._name = name
        self.squeeze_conv = tf.keras.layers.Conv2D(filters=C_squeeze, kernel_size=1, strides=1,
                                                   padding="same", data_format=data_format,
                                                   activation=None, use_bias=True,
                                                   kernel_initializer="he_normal",
                                                   name=self._name + "/s_conv")
        self.act = tf.keras.layers.ReLU()
        self.excite_conv = tf.keras.layers.Conv2D(filters=C_in, kernel_size=1, strides=1,
                                                  padding="same", data_format=data_format,
                                                  activation=None, use_bias=True,
                                                  kernel_initializer="he_normal",
                                                  name=self._name + "/e_conv")
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.spatial_dims = [2, 3]
        else:
            self.channel_axis = 3
            self.spatial_dims = [1, 2]
        self.h_sigmoid = HSigmoid()

    def call(self, inputs, training=True):
        with tf.name_scope(self._name):
            x = tf.reduce_mean(inputs, self.spatial_dims, keepdims=True)
            x = self.squeeze_conv(x)
            x = self.act.call(x)
            x = self.excite_conv(x)
        return self.h_sigmoid.call(x) * inputs


class MBConv2(tf.keras.layers.Layer):

    def __init__(self, C_in, C_out, kernel_size, stride,
                 name="MBConv2", data_format="channels_last",
                 expansion_ratio=6, enable_skip=True):
        super(MBConv2, self).__init__()
        self._name = name
        self.enable_skip = enable_skip and stride == 1 and C_in == C_out
        self.expanded_channel_size = int(C_in * expansion_ratio)
        if data_format == "channels_last":
            self.channel_axis = 3
        else:
            self.channel_axis = 1
        self.expansion_conv = None
        if expansion_ratio > 1:
            self.expansion_conv = AugmentedConv2D(filters=self.expanded_channel_size,
                                                  kernel_size=1, stride=1, padding="same",
                                                  activation=tf.keras.layers.ReLU(max_value=6),
                                                  data_format=data_format,
                                                  name=self._name + "/expand_conv")
        self.depth_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride,
                                                          padding="same", data_format=data_format,
                                                          name=self._name + "/depth_conv", use_bias=False,
                                                          kernel_initializer="he_normal")
        self.depth_bn = tf.keras.layers.BatchNormalization(axis=self.channel_axis, trainable=True,
                                                           momentum=BN_MOMENTUM, epsilon=BN_EPSILON,
                                                           name=self._name + "/depth_bn")
        self.depth_relu = tf.keras.layers.ReLU(max_value=6)
        self.point_conv = AugmentedConv2D(filters=C_out, kernel_size=1, stride=1, padding="same",
                                          activation=None, data_format=data_format,
                                          name=self._name + "/point_conv")

    def call(self, inputs, training=True):
        with tf.name_scope(self._name):
            if self.expansion_conv is not None:
                x = self.expansion_conv(inputs, training=training)
            else:
                x = inputs
            x = self.depth_conv(x)
            x = self.depth_bn(x, training=training)
            x = self.depth_relu(x)
            x = self.point_conv(x, training=training)
            if self.enable_skip:
                x = x + inputs
        return x


class MBConv3(tf.keras.layers.Layer):

    def __init__(self, C_in, C_out, kernel_size, stride,
                 name="MBConv3", data_format="channels_last",
                 expansion_ratio=6, se_ratio=0.25, enable_skip=True,
                 act_type="relu"):
        super(MBConv3, self).__init__()
        self._name = name
        self.act_type = act_type
        self.enable_skip = enable_skip and stride == 1 and C_in == C_out
        self.expanded_channel_size = int(C_in * expansion_ratio)
        self.has_se = 1 >= se_ratio > 0
        self.se_channel_size = get_final_channel_size(max(1, int(self.expanded_channel_size * se_ratio)), 1.0)
        if data_format == "channels_last":
            self.channel_axis = 3
        else:
            self.channel_axis = 1
        self.expansion_conv = None
        if expansion_ratio > 1:
            self.expansion_conv = AugmentedConv2D(filters=self.expanded_channel_size,
                                                  kernel_size=1, stride=1, padding="same",
                                                  activation=HSwish() if self.act_type=="swish" else tf.keras.layers.ReLU(),
                                                  data_format=data_format,
                                                  name=self._name + "/expand_conv")
        self.depth_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=stride,
                                                          padding="same", data_format=data_format,
                                                          name=self._name + "/depth_conv", use_bias=False,
                                                          kernel_initializer="he_normal")
        self.depth_bn = tf.keras.layers.BatchNormalization(axis=self.channel_axis, trainable=True,
                                                           momentum=BN_MOMENTUM, epsilon=BN_EPSILON,
                                                           name=self._name + "/depth_bn")
        self.depth_act = HSwish() if self.act_type=="swish" else tf.keras.layers.ReLU()
        self.se_module = None
        if self.has_se:
            self.se_module = SqueezeExcite(self.expanded_channel_size, self.se_channel_size,
                                           name=self._name + "/se", data_format=data_format)
        self.point_conv = AugmentedConv2D(filters=C_out, kernel_size=1, stride=1, padding="same",
                                          activation=None, data_format=data_format,
                                          name=self._name + "/point_conv")

    def call(self, inputs, training=True):
        with tf.name_scope(self._name):
            if self.expansion_conv is not None:
                x = self.expansion_conv(inputs, training=training)
            else:
                x = inputs
            x = self.depth_conv(x)
            x = self.depth_bn(x, training=training)
            x = self.depth_act(x)
            if self.has_se:
                x = self.se_module(x, training=training)
            x = self.point_conv(x, training=training)
            if self.enable_skip:
                x = x + inputs
        return x


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, conv, shortcut,
                 name="ResidualBlock"):
        super(ResidualBlock, self).__init__()
        self._name = name
        self.conv = conv
        self.shortcut = shortcut

    def call(self, x, training=True):
        with tf.name_scope(self._name):
            if self.conv is None:
                res = x
            elif self.shortcut is None:
                res = self.conv.call(x, training=training)
            else:
                res = self.conv(x, training=training) + self.shortcut(x, training=training)
            return res


class Identity(tf.keras.layers.Layer):

    def __init__(self, name="Identity", **kwargs):
        super(Identity, self).__init__()
        self._name = name

    def call(self, x, **kwargs):
        return x


class ResNetBlock(tf.keras.layers.Layer):

    def __init__(self, C_in, C_out, kernel_size, stride,
                 expansion_ratio=0.25, downsample_mode="avgpool_conv",
                 name="ResNetBlock", data_format="channels_last"):
        super(ResNetBlock, self).__init__()
        self._name = name
        self.data_format = data_format
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        self.stride = stride
        self.expansion_ratio = expansion_ratio
        self.downsample_mode = downsample_mode
        self.C_mid = make_divisible(round(C_out * expansion_ratio), 8)
        self.conv1 = AugmentedConv2D(filters=self.C_mid,
                                     kernel_size=1, stride=1, padding="same",
                                     activation=tf.keras.layers.ReLU(),
                                     data_format=data_format,
                                     use_bn=True, use_bias=False,
                                     name=self._name + "/conv1")
        self.conv2 = AugmentedConv2D(filters=self.C_mid,
                                     kernel_size=3, stride=stride, padding="same",
                                     activation=tf.keras.layers.ReLU(),
                                     data_format=data_format,
                                     use_bn=True, use_bias=False,
                                     name=self._name + "/conv2")
        self.conv3 = AugmentedConv2D(filters=self.C_out,
                                     kernel_size=1, stride=1, padding="same",
                                     activation=None,
                                     data_format=data_format,
                                     use_bn=True, use_bias=False,
                                     name=self._name + "/conv3")
        if self.stride == 1 and self.C_in == self.C_out:
            self.downsample = Identity(self._name + "/downsample")
        elif self.downsample_mode == 'conv':
            self.downsample = AugmentedConv2D(filters=self.C_out,
                                              kernel_size=1, stride=1, padding="same",
                                              activation=None,
                                              data_format=data_format,
                                              use_bn=True, use_bias=False,
                                              name=self._name + "/downsample")
        elif self.downsample_mode == 'avgpool_conv':
            self.downsample = tf.keras.Sequential([
                tf.keras.layers.AveragePooling2D(pool_size=stride, strides=stride, padding="same",
                                                 data_format=self.data_format,
                                                 name=self._name + "/downsample/avgpool2"),
                AugmentedConv2D(filters=self.C_out,
                                kernel_size=1, stride=1, padding="same",
                                activation=None,
                                data_format=data_format,
                                use_bn=True, use_bias=False,
                                name=self._name + "/downsample/conv")
            ])
        else:
            raise NotImplementedError
        self.final_act = tf.keras.layers.ReLU(name=self._name + "/final_act")

    def call(self, x, training=True):
        res = self.downsample(x, training=training)
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        x = x + res
        x = self.final_act(x)
        return x
