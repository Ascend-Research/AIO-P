import torch as t
import tensorflow.compat.v1 as tf
from model_src.model_components import Identity
import numpy as np


class TaskAdapterBase(t.nn.Module):

    def __init__(self, name="BaseAdapter"):
        super(TaskAdapterBase, self).__init__()
        self.name = name  

    def build(self, resolution=None):
        self.input_res = resolution
        self.op = Identity()

    def forward(self, x):
        return self.op(x)

    def tf_model_maker(self):
        return TFTaskAdapterBase()


class TFTaskAdapterBase(tf.keras.Model):
    def __init__(self):
        super(TFTaskAdapterBase, self).__init__()
        self.op = tf.identity

    def call(self, x, training=True):
        with tf.name_scope(self._name):
            return self.op(x)


class DeconvHPEHead(TaskAdapterBase):
    def __init__(self, name="BaseAdapter", hw=(64, 64), joints=16):
        super(DeconvHPEHead, self).__init__(name=name)
        self.hw = hw
        self.joints = joints

    def build(self, resolution=(8, 8, 2048), make_rn_conv=False):
        self.input_res = resolution

        self.rn_conv = None
        if make_rn_conv:
            self.rn_conv = t.nn.Sequential(
                t.nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1, stride=1, padding=0),
                t.nn.BatchNorm2d(2048, momentum=0.1),
                t.nn.ReLU(inplace=True)
            )

        self.num_deconv_layers = int(np.log2(self.hw[0] // self.input_res[0]))
        self.torch_deconv = self._make_deconv_layer(self.num_deconv_layers, [256] * self.num_deconv_layers,
                                                    [4] * self.num_deconv_layers)
        self.torch_final_layer = t.nn.Conv2d(
            in_channels=256 if self.num_deconv_layers > 0 else self.input_res[-1],
            out_channels=self.joints,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        self.filters, self.kernels = num_filters, num_kernels
        for i in range(num_layers):
            padding, output_padding = get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                t.nn.ConvTranspose2d(
                    in_channels=num_filters[i] if i > 0 else self.input_res[-1],
                    out_channels=num_filters[i],
                    kernel_size=num_kernels[i],
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False)) 
            layers.append(t.nn.BatchNorm2d(planes, momentum=0.1))
            layers.append(t.nn.ReLU(inplace=True))

        return t.nn.Sequential(*layers)

    def forward(self, x):
        x = x[-1]
        if self.rn_conv is not None:
            x = self.rn_conv(x)
        x = self.torch_deconv(x)
        return self.torch_final_layer(x)

    def tf_model_maker(self):
        return TFDeconvHPEHead(self.num_deconv_layers, self.filters, self.kernels, self.joints, self.rn_conv)


class TFDeconvHPEHead(tf.keras.Model):
    def __init__(self, num_layers, filters, kernels, joints=16, make_rn_conv=None):
        super(TFDeconvHPEHead, self).__init__()
        assert len(filters) == num_layers
        assert len(kernels) == num_layers

        layers = []
        if make_rn_conv is not None:
            layers.append(tf.keras.layers.Conv2D(filters=2048, kernel_size=1, strides=1, padding="same"))
            layers.append(tf.keras.layers.BatchNormalization(axis=3, trainable=True, momentum=0.1))
            layers.append(tf.keras.layers.ReLU())

        for i in range(num_layers):
            layers.append(
                tf.keras.layers.Conv2DTranspose(
                    filters=filters[i],
                    kernel_size=kernels[i],
                    strides=2,
                    padding="same",
                    use_bias=False))
            layers.append(tf.keras.layers.BatchNormalization(axis=3, trainable=True, momentum=0.1))
            layers.append(tf.keras.layers.ReLU())
        layers.append(
            tf.keras.layers.Conv2D(
                filters=joints,
                kernel_size=1,
                strides=1,
                padding="same"))
        self.sequence = tf.keras.Sequential(layers)

    def call(self, x, training=True):
        x = x[-1]
        with tf.name_scope(self._name):
            return self.sequence(x, training=training)


def get_deconv_cfg(deconv_kernel):
    if deconv_kernel == 4:
        padding = 1
        output_padding = 0
    elif deconv_kernel == 3:
        padding = 1
        output_padding = 1
    elif deconv_kernel == 2:
        padding = 0
        output_padding = 0

    return padding, output_padding