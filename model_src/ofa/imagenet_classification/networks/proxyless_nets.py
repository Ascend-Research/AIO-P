# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import torch.nn as nn
from model_src.ofa.utils.layers import set_layer_from_config, MBConvLayer, IdentityLayer,  ResidualBlock
from model_src.ofa.utils import MyNetwork, MyGlobalAvgPool2d

__all__ = ['ProxylessNASNets']

class ProxylessNASNets(MyNetwork):

	def __init__(self, first_conv, blocks, feature_mix_layer, classifier):
		super(ProxylessNASNets, self).__init__()

		self.first_conv = first_conv
		self.blocks = nn.ModuleList(blocks)
		self.feature_mix_layer = feature_mix_layer
		self.global_avg_pool = MyGlobalAvgPool2d(keep_dim=False)
		self.classifier = classifier

	def forward(self, x):
		x = self.first_conv(x)
		for block in self.blocks:
			x = block(x)
		if self.feature_mix_layer is not None:
			x = self.feature_mix_layer(x)
		x = self.global_avg_pool(x)
		x = self.classifier(x)
		return x

	@property
	def module_str(self):
		_str = self.first_conv.module_str + '\n'
		for block in self.blocks:
			_str += block.module_str + '\n'
		_str += self.feature_mix_layer.module_str + '\n'
		_str += self.global_avg_pool.__repr__() + '\n'
		_str += self.classifier.module_str
		return _str

	@property
	def config(self):
		return {
			'name': ProxylessNASNets.__name__,
			'bn': self.get_bn_param(),
			'first_conv': self.first_conv.config,
			'blocks': [
				block.config for block in self.blocks
			],
			'feature_mix_layer': None if self.feature_mix_layer is None else self.feature_mix_layer.config,
			'classifier': self.classifier.config,
		}

	@staticmethod
	def build_from_config(config):
		first_conv = set_layer_from_config(config['first_conv'])
		feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
		classifier = set_layer_from_config(config['classifier'])

		blocks = []
		for block_config in config['blocks']:
			blocks.append(ResidualBlock.build_from_config(block_config))

		net = ProxylessNASNets(first_conv, blocks, feature_mix_layer, classifier)
		if 'bn' in config:
			net.set_bn_param(**config['bn'])
		else:
			net.set_bn_param(momentum=0.1, eps=1e-3)

		return net

	def zero_last_gamma(self):
		for m in self.modules():
			if isinstance(m, ResidualBlock):
				if isinstance(m.conv, MBConvLayer) and isinstance(m.shortcut, IdentityLayer):
					m.conv.point_linear.bn.weight.data.zero_()

	@property
	def grouped_block_index(self):
		info_list = []
		block_index_list = []
		for i, block in enumerate(self.blocks[1:], 1):
			if block.shortcut is None and len(block_index_list) > 0:
				info_list.append(block_index_list)
				block_index_list = []
			block_index_list.append(i)
		if len(block_index_list) > 0:
			info_list.append(block_index_list)
		return info_list

	def load_state_dict(self, state_dict, **kwargs):
		current_state_dict = self.state_dict()

		for key in state_dict:
			if key not in current_state_dict:
				assert '.mobile_inverted_conv.' in key
				new_key = key.replace('.mobile_inverted_conv.', '.conv.')
			else:
				new_key = key
			current_state_dict[new_key] = state_dict[key]
		super(ProxylessNASNets, self).load_state_dict(current_state_dict)


