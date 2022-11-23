from model_src.multitask.fpn_adapter_base import FeaturePyramidAdapterBase
import torch as t
import fvcore.nn.weight_init as weight_init


class FeaturePyramidDetectron(FeaturePyramidAdapterBase):
    def __init__(self, name="BaseAdapter", hw=(256, 256), add_downsample=0):
        super(FeaturePyramidDetectron, self).__init__(name=name, uniform_channels=True, hw=hw, add_downsample=add_downsample)
        self.hw = hw

    def build(self, resolution=(8, 8, 2048), make_rn_conv=False, skip_dims=None):
        super().build(resolution=resolution, make_rn_conv=make_rn_conv, skip_dims=skip_dims)

        weight_init.c2_xavier_fill(self.rn_conv)

        for skip_conv in self.fpn_convs:
            weight_init.c2_xavier_fill(skip_conv)

        for deconv in self.torch_deconv:
            weight_init.c2_xavier_fill(deconv)

        for downsample in self.downsample_blks:
            weight_init.c2_xavier_fill(downsample)

    def forward(self, x):
        x_skip, x = x[:-1], x[-1]
        x_skip = x_skip[::-1]
        x_fpn = []

        x = self.rn_conv(x)
        x_ds = x
        for ds_blk in self.downsample_blks:
            x_ds = ds_blk(x_ds)
            x_fpn.append(x_ds)
        x_fpn = x_fpn[::-1]
        for i, deconv_block in enumerate(self.torch_deconv):
            x_fpn.append(x)
            x = deconv_block(x)

            concat_list = []
            for j in self.skip_inds[i]:
                skip_tensor = x_skip[j]
                concat_list.append(skip_tensor)
            if len(concat_list) > 1:
                skip_tensor = t.cat(concat_list, dim=1)
            skip_tensor = self.fpn_convs[i](skip_tensor)
            x = x + skip_tensor
        x_fpn.append(x)
        return x_fpn
