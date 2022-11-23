from model_src.multitask.task_adapter import TFTaskAdapterBase
from model_src.multitask.fpn_adapter_base import FeaturePyramidAdapterBase
import tensorflow.compat.v1 as tf
import torch as t


class SkipDeconvHPEHead(FeaturePyramidAdapterBase):
    def __init__(self, name="BaseAdapter", hw=(64, 64), joints=16):
        super(SkipDeconvHPEHead, self).__init__(uniform_channels=False, name=name, hw=hw, add_downsample=0)
        self.joints = joints

    def build(self, resolution=(8, 8, 2048), make_rn_conv=False, skip_dims=None):
        super().build(resolution=resolution, make_rn_conv=make_rn_conv, skip_dims=skip_dims)

        if make_rn_conv:
            rn_conv = self.rn_conv
            self.rn_conv = t.nn.Sequential(rn_conv,
                           t.nn.BatchNorm2d(rn_conv.out_channels, momentum=0.1),
                           t.nn.ReLU(inplace=True))

        for i, fpn_conv in enumerate(self.fpn_convs):
            self.fpn_convs[i] = t.nn.Sequential(fpn_conv,
                                t.nn.BatchNorm2d(fpn_conv.out_channels, momentum=0.1),
                                t.nn.ReLU(inplace=True))

        for i, deconv in enumerate(self.torch_deconv):
            self.torch_deconv[i] = t.nn.Sequential(deconv,
                                   t.nn.BatchNorm2d(deconv.out_channels, momentum=0.1),
                                   t.nn.ReLU(inplace=True))

        final_input_channels = 256 if self.num_deconv_layers > 0 else self.input_res[-1]
        self.torch_final_layer = t.nn.Conv2d(
            in_channels=final_input_channels,
            out_channels=self.joints,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        x_fpn = super().forward(x)

        return self.torch_final_layer(x_fpn[-1])

    def tf_model_maker(self):
        return TFSkipDeconvHPEHead(self.num_deconv_layers, self.filters, self.kernels, self.joints, self.rn_conv,
                                   skip_inds=self.skip_inds)

class TFSkipDeconvHPEHead(TFTaskAdapterBase):
    def __init__(self, num_layers, filters, kernels, joints=16, make_rn_conv=None, skip_inds=None):
        super(TFSkipDeconvHPEHead, self).__init__()
        assert len(filters) == num_layers
        assert len(kernels) == num_layers

        self.skip_inds = skip_inds

        self.deconv_blocks = []
        self.rn_conv = None
        if make_rn_conv is not None:
            self.rn_conv = tf.keras.Sequential()
            self.rn_conv.add(tf.keras.layers.Conv2D(filters=2048, kernel_size=1, strides=1, padding="same",
                                                    use_bias=False))
            self.rn_conv.add(tf.keras.layers.BatchNormalization(axis=3, trainable=True, momentum=0.1))
            self.rn_conv.add(tf.keras.layers.ReLU())

        self.fpn_convs = []
        for _ in skip_inds:
            skip_conv = tf.keras.Sequential()
            skip_conv.add(tf.keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding="same",
                                                 use_bias=False))
            skip_conv.add(tf.keras.layers.BatchNormalization(axis=3, trainable=True, momentum=0.1))
            skip_conv.add(tf.keras.layers.ReLU())
            self.fpn_convs.append(skip_conv)

        for i in range(num_layers):
            layers = tf.keras.Sequential()
            layers.add(
                tf.keras.layers.Conv2DTranspose(
                    filters=filters[i],
                    kernel_size=kernels[i],
                    strides=2,
                    padding="same",
                    use_bias=False))
            layers.add(tf.keras.layers.BatchNormalization(axis=3, trainable=True, momentum=0.1))
            layers.add(tf.keras.layers.ReLU())
            self.deconv_blocks.append(layers)

        self.final_conv = tf.keras.layers.Conv2D(
            filters=joints,
            kernel_size=1,
            strides=1,
            padding="same")

    def call(self, x, training=True):
        x_skip, x = x[:-1], x[-1]
        x_skip = x_skip[::-1]
        x_fpn = []
        with tf.name_scope(self._name):
            if self.rn_conv is not None:
                x = self.rn_conv(x, training=training)
            for i, deconv_block in enumerate(self.deconv_blocks):
                x_fpn.append(x)
                x = deconv_block(x, training=training)
                if len(self.skip_inds) > 0:
                    concat_list = []
                    for j in self.skip_inds[i]:
                        skip_tensor = x_skip[j]
                        concat_list.append(skip_tensor)
                    if len(concat_list) > 1:
                        skip_tensor = tf.keras.layers.Concatenate(axis=-1)(concat_list)
                    skip_tensor = self.fpn_convs[i](skip_tensor, training=training)
                    x = x + skip_tensor
            x_fpn.append(x)
            return self.final_conv(x_fpn[-1])
