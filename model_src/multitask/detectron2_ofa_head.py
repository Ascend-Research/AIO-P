from model_src.ofa.model_zoo import ofa_net
from model_src.multitask.adapt_ofa_head import OFAAdaptedCGHead
from model_src.ofa.my_utils import ofa_str_configs_to_subnet_args


class Detectron2OFAHead(OFAAdaptedCGHead):

    def __init__(self, base_cg, original_config, task_head, input_dims=[1024, 1024, 3], freeze=2):
        self.freeze_level = freeze
        super(Detectron2OFAHead, self).__init__(base_cg=base_cg, original_config=original_config, 
                                                task_head=task_head, input_dims=input_dims,
                                                swap_num=-1, backprop=False, skip=True)

    def _compile_network(self):
        self._calc_nodes_edges()

        if "mbv3" in self._extract_cg_name().lower():
            supernet = ofa_net("ofa_mbv3_d234_e346_k357_w1.2", pretrained=True)
            arch_config = self.original_config[-1]
            ks, es, ds = ofa_str_configs_to_subnet_args(arch_config, 20,
            fill_k=3, fill_e=3, expected_prefix="mbconv3")
            supernet.set_active_subnet(ks=ks, e=es, d=ds)
            self.body_model = supernet.get_active_subnet()
            self.body = self.forward_mbv3
            if self.freeze_level > 0:
                for param in self.body_model.first_conv.parameters():
                    param.requires_grad = False
        elif "resnet" in self._extract_cg_name().lower():
            supernet = ofa_net("ofa_resnet50", pretrained=True)
            arch_config = self.original_config[-1]
            supernet.set_active_subnet(d=arch_config['d'],
            e=arch_config['e'], w=arch_config['w'])
            self.body_model = supernet.get_active_subnet()
            self.body = self.forward_resnet
            if self.freeze_level > 0:
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
            self.body = self.forward_pn
            if self.freeze_level > 0:
                for param in self.body_model.first_conv.parameters():
                    param.requires_grad = False

        self.supernet = supernet
        self.ds_list, ds_dims = self._get_block_output_channels(get_hw=True)
        self.task_head.build(resolution=self.final_res, make_rn_conv=False, skip_dims=ds_dims) 
            
        if self.freeze_level > 1:
            final_blk_to_freeze = self.ds_list[self.freeze_level - 1]
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
