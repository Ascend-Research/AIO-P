import torch as t
import torch.nn as nn


class AdaptedSamplingHead(nn.Module):
    def __init__(self, base_cg, task_head, input_dims=[224, 224, 3], skip=False, latent_processor=None):
        super().__init__()
        self.skip = skip
        self.task_head = task_head
        self.latent_processor = latent_processor
        self.forward = self.forward_processor if latent_processor is not None else self.forward_cache
        family = base_cg.name.lower()
        rn_conv = False
        if "mbv3" in family:
            latent_channels = 1152
        elif "resnet" in family:
            latent_channels = 2048
            rn_conv = True
        else: #PN
            latent_channels = 1664
        self.final_res = (
            input_dims[0] // 32,
            input_dims[1] // 32,
            latent_channels,
        )
        self.input_dims = input_dims

        if self.skip:
            skip_dims = self._build_skip_dims(family, input_dims)
            self.task_head.build(
                resolution=self.final_res,
                make_rn_conv=rn_conv,
                skip_dims=skip_dims, 
            )
        else:
            self.task_head.build(resolution=self.final_res, make_rn_conv=rn_conv)

    def _build_skip_dims(self, cg_name, input_dims):
        skip_dims = []
        input_dims = input_dims.copy()
        input_dims[0] //= 2
        input_dims[1] //= 2

        if "resnet" in cg_name:
            latent_channels = [256, 512, 1048]
            stage_strides = [2, 2, 2]
        else:
            stage_strides = [2, 2, 2, 1]
            if "mbv3" in cg_name:
                latent_channels = [32, 48, 96, 136]
            else: # PN
                latent_channels = [32, 56, 104, 128]

        for i in range(len(latent_channels)): 
            input_dims[0] //= stage_strides[i]
            input_dims[1] //= stage_strides[i]
            skip_dims.append([latent_channels[i], input_dims[0], input_dims[1]])
        return skip_dims

    def set_optim_params(self):
        return None

    def build_overall_cg(self):
        return None

    def forward_cache(self, x):
        return self.task_head([x])

    def forward_processor(self, x):
        latent_list = self.latent_processor(x)

        x_list = []
        for (mean, std_dev) in latent_list:
            zeta = t.randn_like(std_dev)
            x_list.append(mean + zeta * std_dev)
        return self.task_head(x_list)
