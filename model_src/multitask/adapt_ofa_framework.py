from model_src.ofa.model_zoo import ofa_net
from model_src.multitask.adapt_cg_framework import TaskAdaptedCGModel
from model_src.ofa.my_utils import ofa_str_configs_to_subnet_args


class OFAAdaptedCGModel(TaskAdaptedCGModel):

    def __init__(self, base_cg, original_config, task_head, input_dims=[224, 224, 3]):
        if "ofa-" not in base_cg.name.lower():
            raise ValueError("Compute Graph is not from OFA. Use TaskAdaptedCGModel instead")
        self.original_config = original_config
        self.body, self.body_model = None, None
        super(OFAAdaptedCGModel, self).__init__(
            base_cg=base_cg,
            task_head=task_head,
            input_dims=input_dims
        )

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
        elif "resnet" in self._extract_cg_name().lower():
            supernet = ofa_net("ofa_resnet50", pretrained=True)
            arch_config = self.original_config[-1]
            supernet.set_active_subnet(d=arch_config['d'],
            e=arch_config['e'], w=arch_config['w'])
            self.body_model = supernet.get_active_subnet()
            self.body = self.forward_resnet
        else:
            arch_config = self.original_config[-1]
            supernet = ofa_net("ofa_proxyless_d234_e346_k357_w1.3", pretrained=True)
            ks, es, ds = ofa_str_configs_to_subnet_args(arch_config, 21, 
            fill_k=3, fill_e=3, expected_prefix="mbconv2")
            supernet.set_active_subnet(ks=ks, e=es, d=ds)
            self.body_model = supernet.get_active_subnet()
            self.body = self.forward_pn

        self.task_head.build(resolution=self.final_res)

    def set_optim_params(self):
        return None

    # Removed global avg pool and classifier
    def forward_resnet(self, x):
        for layer in self.body_model.input_stem:
            x = layer(x)
        x = self.body_model.max_pooling(x)
        for block in self.body_model.blocks:
            x = block(x)
        return [x]

    # Removed global avg pool and classifier
    def forward_pn(self, x):
        x = self.body_model.first_conv(x)
        for block in self.body_model.blocks:
            x = block(x)
        if self.body_model.feature_mix_layer is not None:
            x = self.body_model.feature_mix_layer(x)
        return [x]

    # Removed global avg pool, feature mix and classifier layers
    def forward_mbv3(self, x):
        x = self.body_model.first_conv(x)
        for block in self.body_model.blocks:
            x = block(x)
        x = self.body_model.final_expand_layer(x)
        return [x]
