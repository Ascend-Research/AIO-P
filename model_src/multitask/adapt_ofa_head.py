import torch
from model_src.ofa.model_zoo import ofa_net
from model_src.multitask.adapt_cg_framework import TaskAdaptedCGModel, TensorFlowTaskModel
from model_src.ofa.my_utils import ofa_str_configs_to_subnet_args


class OFAAdaptedCGHead(TaskAdaptedCGModel):

    def __init__(self, base_cg, original_config, task_head, input_dims=[224, 224, 3], swap_num=-1, backprop=False, skip=True):
        if "ofa-" not in base_cg.name.lower():
            raise ValueError("Compute Graph is not from OFA. Use TaskAdaptedCGModel instead")
        self.original_config = original_config
        self.supernet = None
        self.optim = None
        self.swap_num, self.swap_ctr = swap_num, 0
        self.backprop = backprop
        self.body_fwd, self.body_model = None, None
        self.skip = skip
        super(OFAAdaptedCGHead, self).__init__(
            base_cg=base_cg,
            task_head=task_head,
            input_dims=input_dims
        )

    def set_optim_params(self):
        if self.swap_num >= 1 and self.backprop == True:
            self.optim.param_groups.clear()
            self.optim.state.clear()
            self.optim.add_param_group({'params': self.task_head.parameters()})
            self.optim.add_param_group({'params': self.body_model.parameters()})

    def _compile_network(self):
        self._calc_nodes_edges()

        make_rn_conv = False
        if "mbv3" in self._extract_cg_name().lower():
            supernet = ofa_net("ofa_mbv3_d234_e346_k357_w1.2", pretrained=True)
            arch_config = self.original_config[-1]
            ks, es, ds = ofa_str_configs_to_subnet_args(arch_config, 20,
            fill_k=3, fill_e=3, expected_prefix="mbconv3")
            supernet.set_active_subnet(ks=ks, e=es, d=ds)
            self.body_model = supernet.get_active_subnet()
            self.body_fwd = self.forward_mbv3
            if isinstance(self.backprop, int) and self.backprop > 0:
                for param in self.body_model.first_conv.parameters():
                    param.requires_grad = False
        elif "resnet" in self._extract_cg_name().lower():
            supernet = ofa_net("ofa_resnet50", pretrained=True)
            supernet.set_max_net()
            self.body_model = supernet.get_active_subnet()
            self.body_fwd = self.forward_resnet
            fr = list(self.final_res)
            fr[-1] = 2048
            self.final_res = tuple(fr)
            make_rn_conv = True
            if isinstance(self.backprop, int) and self.backprop > 0:
                for param in self.body_model.input_stem.parameters():
                    param.requires_grad = False
                for param in self.body_model.max_pooling.parameters():
                    param.requires_grad = False
        else:
            arch_config = self.original_config[-1]
            supernet = ofa_net("ofa_proxyless_d234_e346_k357_w1.3", pretrained=True)
            ks, es, ds = ofa_str_configs_to_subnet_args(arch_config, 21,
            fill_k=3, fill_e=3, expected_prefix="mbconv2")
            supernet.set_active_subnet(ks=ks, e=es, d=ds)
            self.body_model = supernet.get_active_subnet()
            self.body_fwd = self.forward_pn
            if isinstance(self.backprop, int) and self.backprop > 0:
                for param in self.body_model.first_conv.parameters():
                    param.requires_grad = False


        self.supernet = supernet
        self.ds_list, ds_dims = [], None
        if self.skip:
            self.ds_list, ds_dims = self._get_block_output_channels(get_hw=True)
            self.task_head.build(resolution=self.final_res, make_rn_conv=make_rn_conv, skip_dims=ds_dims)
        else:
            self.task_head.build(resolution=self.final_res, make_rn_conv=make_rn_conv)
        
        if make_rn_conv:
            arch_config = self.original_config[-1]
            supernet.set_active_subnet(d=arch_config['d'], e=arch_config['e'], w=arch_config['w'])
            self.body_model = supernet.get_active_subnet()
            self.ds_list = self._get_block_output_channels()

        if self.swap_num >= 1 and self.backprop == False:
            for param in self.body_model.parameters():
                param.requires_grad = False
        elif isinstance(self.backprop, int) and self.backprop > 1 and self.swap_num < 0:
            final_blk_to_freeze = self.ds_list[self.backprop - 1]
            for block in self.body_model.blocks[:final_blk_to_freeze + 1]:
                for param in block.parameters():
                    param.requires_grad = False


    # Removed global avg pool and classifier
    def forward_resnet(self, x):
        x_list = []
        for layer in self.body_model.input_stem:
            x = layer(x)
        x = self.body_model.max_pooling(x)
        for i, block in enumerate(self.body_model.blocks):
            x = block(x)
            if i in self.ds_list:
                x_list.append(x)
        # Added to accomodate variable unit-wide channel ratios
        if x.shape[1] < 2048:
            channel_deficit = 2048 - x.shape[1]
            zero_padding = torch.zeros([x.shape[0], channel_deficit,
                                        x.shape[2], x.shape[3]])
            x = torch.cat([x, zero_padding.to(x.device)], dim=1)
        x_list.append(x)
        return x_list

    # Removed global avg pool and classifier
    def forward_pn(self, x):
        x_list = []
        x = self.body_model.first_conv(x)
        for i, block in enumerate(self.body_model.blocks):
            x = block(x)
            if i in self.ds_list:
                x_list.append(x)
        if self.body_model.feature_mix_layer is not None:
            x = self.body_model.feature_mix_layer(x)
        x_list.append(x)
        return x_list

    # Removed global avg pool, feature mix and classifier layers
    def forward_mbv3(self, x):
        x_list = []
        x = self.body_model.first_conv(x)
        for i, block in enumerate(self.body_model.blocks):
            x = block(x)
            if i in self.ds_list:
                x_list.append(x)
        x = self.body_model.final_expand_layer(x)
        x_list.append(x)
        return x_list

    # New forward function wrapper
    def body(self, x):
        if self.swap_num >= 1:
            self.swap_ctr += 1
        if self.swap_ctr == self.swap_num:
            self.swap_ctr = 0
            # Swap to new subnet.
            new_subnet_config = self.supernet.sample_active_subnet()
            print("Swapping to new subnet: ", new_subnet_config)
            self.body_model = self.supernet.get_active_subnet()
            self.ds_list = self._get_block_output_channels()
            self.set_optim_params()
            
            if self.swap_num >= 1 and self.backprop == False:
                for param in self.body_model.parameters():
                    param.requires_grad = False

        return self.body_fwd(x)

    def _get_block_output_channels(self, get_hw=False):
        out_channel_list = []
        downsample_list = []
        eval_str = "conv.point_linear.conv"
        if "resnet" in self._extract_cg_name().lower():
            eval_str = "conv3.conv"
        for i, block in enumerate(self.body_model.blocks[1:]):
            out_channel_list.append(eval("block.{}.out_channels".format(eval_str)))
            if len(out_channel_list) > 1 and out_channel_list[-1] != out_channel_list[-2]:
                downsample_list.append(i)
        if "pn" in self._extract_cg_name().lower():
            out_channel_list = out_channel_list[:-1]
            downsample_list = downsample_list[:-1]

        if get_hw:
            chw_list = []
            with torch.no_grad():
                [h, w, c] = self.input_dims
                test_tensor = torch.randn([1, c, h, w])
                if "resnet" in self._extract_cg_name().lower():
                    for layer in self.body_model.input_stem:
                        test_tensor = layer(test_tensor)
                    test_tensor = self.body_model.max_pooling(test_tensor)
                else:
                    test_tensor = self.body_model.first_conv(test_tensor)
                for i, block in enumerate(self.body_model.blocks):
                    test_tensor = block(test_tensor)
                    if i in downsample_list:
                        chw_list.append(list(test_tensor.shape)[1:])
            return downsample_list, chw_list
        return downsample_list

    # We need a special function for the skip-connection head
    def tf_model_maker(self):
        name = self._extract_cg_name().lower()
        if "mbv3" in name:
            from model_src.search_space.ofa_profile.networks_tf import OFAMbv3Net
            tf_body = OFAMbv3Net(self.original_config[-1], skip_list=self.ds_list)
        elif "pn" in name:
            from model_src.search_space.ofa_profile.networks_tf import OFAProxylessNet
            tf_body = OFAProxylessNet(self.original_config[-1], skip_list=self.ds_list)
        elif "resnet" in name:
            from model_src.search_space.ofa_profile.networks_tf import OFAResNet
            tf_body = OFAResNet(self.original_config[-1], skip_list=self.ds_list)
        else:
            raise NotImplementedError
        tf_task_head = self.task_head.tf_model_maker()
        return TensorFlowTaskModel(tf_body, tf_task_head)
