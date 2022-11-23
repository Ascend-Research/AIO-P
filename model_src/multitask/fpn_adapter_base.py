from model_src.multitask.task_adapter import TaskAdapterBase
import tensorflow.compat.v1 as tf
import numpy as np
import torch as t

class FeaturePyramidAdapterBase(TaskAdapterBase):
    def __init__(self, name="BaseAdapter", uniform_channels=False, hw=(64, 64), add_downsample=0):
        super(FeaturePyramidAdapterBase, self).__init__(name=name)
        self.hw = hw
        self.uniform_channels = uniform_channels
        self.add_downsample = add_downsample

    def build(self, resolution=(8, 8, 2048), make_rn_conv=False, skip_dims=None):
        self.input_res = resolution

        self.init_conv_c = 256 if self.uniform_channels else self.input_res[-1]
        self.rn_conv = None
        if make_rn_conv or self.uniform_channels:
            self.rn_conv = t.nn.Conv2d(in_channels=self.input_res[-1], out_channels=self.init_conv_c, kernel_size=1, stride=1, padding=0)

        self.num_deconv_layers = int(np.log2(self.hw[0] // self.input_res[0]))

        self.add_skip_channels = []
        self.skip_inds = []
        if skip_dims is not None:
            skip_dims = skip_dims[::-1]
            curr_h = -1
            for i, latent_shape in enumerate(skip_dims):
                if skip_dims[i][1] > self.hw[0]:
                    break
                elif skip_dims[i][1] != curr_h:
                    curr_h = skip_dims[i][1]
                    self.add_skip_channels.append(skip_dims[i][0])
                    self.skip_inds.append([i])
                else:
                    self.add_skip_channels[-1] += skip_dims[i][0]
                    self.skip_inds[-1].append(i)

        if make_rn_conv:
            self.add_skip_channels = [1024, 512, 256]

        self.fpn_convs = t.nn.ModuleList()
        for channels in self.add_skip_channels:
            skip_conv = t.nn.Conv2d(in_channels=channels, out_channels=256, kernel_size=1, stride=1, padding=0)
            self.fpn_convs.append(skip_conv)

        self.torch_deconv = self._make_deconv_layer(self.num_deconv_layers, [256] * self.num_deconv_layers,
                                                    [4] * self.num_deconv_layers)
    
        # https://arxiv.org/pdf/1708.02002.pdf page 4, they add downsamples in the FPN head
        if self.add_downsample > 0:
            self.downsample_blks = t.nn.ModuleList()
            for _ in range(self.add_downsample):
                ds_blk = t.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
                self.downsample_blks.append(ds_blk)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        deconv_blocks = t.nn.ModuleList()
        self.filters, self.kernels = num_filters, num_kernels
        for i in range(num_layers):
            padding, output_padding = get_deconv_cfg(num_kernels[i])
            deconv = t.nn.ConvTranspose2d(
                        in_channels=num_filters[i] if i > 0 else self.init_conv_c,
                        out_channels=num_filters[i],
                        kernel_size=num_kernels[i],
                        stride=2,
                        padding=padding,
                        output_padding=output_padding,
                        bias=False)
            deconv_blocks.append(deconv)

        return deconv_blocks

    def forward(self, x):
        x_skip, x = x[:-1], x[-1]
        x_skip = x_skip[::-1]
        x_fpn = []

        if self.rn_conv is not None:
            x = self.rn_conv(x)
            if self.add_downsample > 0:
                x_ds = x
                for ds_blk in self.downsample_blks:
                    x_ds = ds_blk(x_ds)
                    x_fpn.append(x_ds)
                x_fpn = x_fpn[::-1]
        for i, deconv_block in enumerate(self.torch_deconv):
            x_fpn.append(x)
            x = deconv_block(x)
            if len(self.skip_inds) > 0:
                concat_list = []
                for j in self.skip_inds[i]:
                    skip_tensor = x_skip[j]
                    concat_list.append(skip_tensor)
                if len(concat_list) > 1:
                    skip_tensor = t.cat(concat_list, dim=1)
                if self.rn_conv is not None:
                        channel_deficit = self.add_skip_channels[i] - skip_tensor.shape[1]
                        if channel_deficit > 0:
                            zero_padding = t.zeros([skip_tensor.shape[0], channel_deficit,
                                                    skip_tensor.shape[2], skip_tensor.shape[3]])
                            skip_tensor = t.cat([skip_tensor, zero_padding.to(skip_tensor.device)], dim=1)
                skip_tensor = self.fpn_convs[i](skip_tensor)
                x = x + skip_tensor
        x_fpn.append(x)
        return x_fpn

    def tf_model_maker(self):
        return TFFeaturePyramid(self.num_deconv_layers, self.filters, self.kernels, self.rn_conv, self.init_conv_c,
                                skip_inds=self.skip_inds, uniform_channels=self.uniform_channels, ds=self.add_downsample)


class TFFeaturePyramid(tf.keras.Model):
    def __init__(self, num_layers, filters, kernels, make_rn_conv=None, init_conv_c=2048, skip_inds=None, uniform_channels=False, ds=0):
        super(TFFeaturePyramid, self).__init__()
        assert len(filters) == num_layers
        assert len(kernels) == num_layers

        self.skip_inds = skip_inds

        self.add_downsample = ds

        self.deconv_blocks = []
        self.rn_conv = None
        if make_rn_conv is not None or uniform_channels:
            self.rn_conv = tf.keras.layers.Conv2D(filters=init_conv_c, kernel_size=1, strides=1, padding="same", use_bias=False)

        self.fpn_convs = []
        for _ in skip_inds:
            skip_conv = tf.keras.layers.Conv2D(filters=256, kernel_size=1, strides=1, padding="same", use_bias=False)
            self.fpn_convs.append(skip_conv)

        for i in range(num_layers):
            deconv = tf.keras.layers.Conv2DTranspose(
                        filters=filters[i],
                        kernel_size=kernels[i],
                        strides=2,
                        padding="same",
                        use_bias=False)
            self.deconv_blocks.append(deconv)

        if ds > 0:
            self.downsample_blks = []
            for _ in range(ds):
                ds_blk = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding="same")
                self.downsample_blks.append(ds_blk)

    def call(self, x, training=True):
        x_skip, x = x[:-1], x[-1]
        x_skip = x_skip[::-1]
        x_fpn = []
        with tf.name_scope(self._name):
            if self.rn_conv is not None:
                x = self.rn_conv(x)
                if self.add_downsample > 0:
                    x_ds = x
                    for ds_blk in self.downsample_blks:
                        x_ds = ds_blk(x_ds)
                        x_fpn.append(x_ds)
                    x_fpn = x_fpn[::-1]
            for i, deconv_block in enumerate(self.deconv_blocks):
                x_fpn.append(x)
                x = deconv_block(x)
                if len(self.skip_inds) > 0:
                    concat_list = []
                    for j in self.skip_inds[i]:
                        skip_tensor = x_skip[j]
                        concat_list.append(skip_tensor)
                    if len(concat_list) > 1:
                        skip_tensor = tf.keras.layers.Concatenate(axis=-1)(concat_list)
                    skip_tensor = self.fpn_convs[i](skip_tensor)
                    x = x + skip_tensor

            x_fpn.append(x)
            return x_fpn


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
