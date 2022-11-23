import torch as t
from model_src.ofa.model_zoo import ofa_net


"""
This script for making a rolling population of architectures and inferring latent representations from them
"""

DEPTH_BINS = [[0, 1],
              [2, 3],
              [4, 6],
              [7, 8],
              [9, 10]]
DEPTH_OFFSETS = {'mbv3': 10,
                 'pn': 11,
                 'resnet': 0}

# OFA architecture configurations described as a dictionary with strings as keys and lists as values. The depth key related to the number of blocks is always 'd'
DEPTH_KEY = 'd'

RESNET_CHANNELS = [256, 512, 1024, 2048]

class ArchSample(t.nn.Module):
    # family - ofa_pn, ofa_mbv3, ofa_resnet
    # n_per_bin, total architectures at a time is this multiplied by len(DEPTH_BINS)
    # use_gpu - for inference
    def __init__(self, family, n_per_bin=5, use_gpu=False, use_logits=True):
        super(ArchSample, self).__init__()

        self.family = family.lower()
        self.gpu = 0 if use_gpu else None
        assert type(n_per_bin) == int
        self.n_per_bin = n_per_bin
        self.num_subnets = n_per_bin * len(DEPTH_BINS)
        self.use_logits = use_logits

        # Get supernets
        # V3
        if "mbv3" in self.family:
            self.family = "mbv3"
            supernet = ofa_net("ofa_mbv3_d234_e346_k357_w1.2", pretrained=True)
            self.depth_key = ''
            self.forward_net = self.forward_mbv3
        # ResNet
        elif "resnet" in self.family:
            self.family = "resnet"
            supernet = ofa_net("ofa_resnet50", pretrained=True)
            self.forward_net = self.forward_resnet
        # ProxylessNAS
        else:
            self.family = "pn"
            supernet = ofa_net("ofa_proxyless_d234_e346_k357_w1.3", pretrained=True)
            self.forward_net = self.forward_pn
        self.supernet = supernet

        self.subnet_list = []
        self._populate_subnet_list()
        # Index used for discarding an architecture and getting a new one.
        self.rrobin_i = 0

    # Initial list of architectures in n_per_bin * len(DEPTH_BINS)
    def _populate_subnet_list(self):
        for _ in range(self.n_per_bin):
            for bin in DEPTH_BINS:
                self.subnet_list.append(self._sample_arch_in_range(range=bin))

    # Range is a sublist from DEPTH_BINs, this samples until something appropriate is found.
    def _sample_arch_in_range(self, range=[0, 1]):
        while True:
            new_config = self.supernet.sample_active_subnet()
            config_len = sum(new_config[DEPTH_KEY]) - DEPTH_OFFSETS[self.family]
            if config_len >= range[0] and config_len <= range[1]:
                subnet = self.supernet.get_active_subnet()
                if self.gpu is not None:
                    subnet = subnet.cuda(device=self.gpu)
                # Return the subnet, its configuration, and calculate block offsets for getting skip tensors
                return [subnet, new_config, self._compute_blk_offsets(new_config)]

    def _compute_blk_offsets(self, config):
        depth_list = config[DEPTH_KEY]
        # Cut out the entry for the stem width
        # Add 2 to all entries because that is the minimum stage size for ResNet
        if self.family == "resnet":
            depth_list = depth_list[1:-1]
            depth_list = [v + 2 for v in depth_list]
            depth_list[-1] += 2
        # Cut out the final 2 stages for PN
        elif self.family == "pn":
            depth_list = depth_list[:-2]
        # Cut out the last stage for MBv3 as we take the final expand instead.
        else:
            depth_list = depth_list[:-1]

        offset_list = [sum(depth_list[:i + 1]) for i, _ in enumerate(depth_list)]
        if self.family == "resnet":
            offset_list = [v - 1 for v in offset_list]
        return offset_list

    # Change 1 architecture with each image
    def _round_robin(self):
        # Pop first element
        del self.subnet_list[0]

        # Select new subnet
        self.subnet_list.append(self._sample_arch_in_range(range=DEPTH_BINS[self.rrobin_i]))

        # update i
        self.rrobin_i += 1
        self.rrobin_i %= len(DEPTH_BINS)

    def forward(self, x):

        if self.gpu is not None:
            if "cuda" in x.device.type and self.gpu != x.device.index:
                self.gpu = int(x.device.index)
                self.subnet_list = []
                self._populate_subnet_list()
            else:
                x = x.cuda(device=self.gpu)

        # Not updating params
        with t.no_grad():
            # Iterate, getting latent representations for each architecture.
            for i, subnet_list in enumerate(self.subnet_list):
                subnet_tensors = self.forward_net(x, subnet_list)
                # This for ResNet, because variable channel sizes.
                subnet_tensors = self.channel_fill(subnet_tensors)

                if i == 0:
                    output_list = [[output_tensor.unsqueeze(0)] for output_tensor in subnet_tensors]
                else:
                    for j in range(len(output_list)):
                        output_list[j].append(subnet_tensors[j].unsqueeze(0))

        # Swap out 1 architecture.
        self._round_robin()
        return output_list

    # Forward function for each family
    def forward_mbv3(self, x, subnet_list):
        [model, _, offsets] = subnet_list
        x = model.first_conv(x)
        x, x_list = self.forward_blocks(x, model.blocks, offsets)
        x = model.final_expand_layer(x)
        x_list.append(x)
        if self.use_logits:
            x = model.global_avg_pool(x)  # global average pooling
            x = model.feature_mix_layer(x)
            x = x.view(x.size(0), -1)
            x = model.classifier(x)
            x_list.append(x)
        return x_list

    def forward_pn(self, x, subnet_list):
        [model, _, offsets] = subnet_list
        x = model.first_conv(x)
        x, x_list = self.forward_blocks(x, model.blocks, offsets)
        if model.feature_mix_layer is not None:
            x = model.feature_mix_layer(x)
        x_list.append(x)
        if self.use_logits:
            x = model.global_avg_pool(x)
            x = model.classifier(x)
            x_list.append(x)
        return x_list

    def forward_resnet(self, x, subnet_list):
        [model, _, offsets] = subnet_list
        for layer in model.input_stem:
            x = layer(x)
        x = model.max_pooling(x)
        x, x_list = self.forward_blocks(x, model.blocks, offsets)
        x_list.append(x)
        if self.use_logits:
            x = model.global_avg_pool(x)
            x = model.classifier(x)
            x_list.append(x)
        return x_list

    def forward_blocks(self, x, blks, offsets):
        x_list = []
        for i, blk in enumerate(blks):
            x = blk(x)
            if i in offsets:
                x_list.append(x)
        return x, x_list

    # This and _zero_pad_c for ResNet
    def channel_fill(self, subnet_tensors):
        if "resnet" not in self.family:
            return subnet_tensors
        for i, max_channels in enumerate(RESNET_CHANNELS):
            subnet_tensors[i] = _zero_pad_c(subnet_tensors[i], max_channels)
        return subnet_tensors

def _zero_pad_c(input, c_max):
    c_deficit = c_max - input.shape[1]
    if c_deficit > 0:
        padding = t.zeros(input.shape[0], c_deficit,
                          input.shape[2], input.shape[3])
        input = t.cat([input, padding.to(input.device)], dim=1)
    return input

