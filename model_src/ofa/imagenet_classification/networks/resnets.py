import torch.nn as nn
from model_src.ofa.utils.layers import set_layer_from_config,  IdentityLayer
from model_src.ofa.utils.layers import ResNetBottleneckBlock
from model_src.ofa.utils import MyNetwork, MyGlobalAvgPool2d

__all__ = ['ResNets']


class ResNets(MyNetwork):

	BASE_DEPTH_LIST = [2, 2, 4, 2]
	STAGE_WIDTH_LIST = [256, 512, 1024, 2048]

	def __init__(self, input_stem, blocks, classifier):
		super(ResNets, self).__init__()

		self.input_stem = nn.ModuleList(input_stem)
		self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
		self.blocks = nn.ModuleList(blocks)
		self.global_avg_pool = MyGlobalAvgPool2d(keep_dim=False)
		self.classifier = classifier

	def forward(self, x):
		for layer in self.input_stem:
			x = layer(x)
		x = self.max_pooling(x)
		for block in self.blocks:
			x = block(x)
		x = self.global_avg_pool(x)
		x = self.classifier(x)
		return x

	@property
	def module_str(self):
		_str = ''
		for layer in self.input_stem:
			_str += layer.module_str + '\n'
		_str += 'max_pooling(ks=3, stride=2)\n'
		for block in self.blocks:
			_str += block.module_str + '\n'
		_str += self.global_avg_pool.__repr__() + '\n'
		_str += self.classifier.module_str
		return _str

	@property
	def config(self):
		return {
			'name': ResNets.__name__,
			'bn': self.get_bn_param(),
			'input_stem': [
				layer.config for layer in self.input_stem
			],
			'blocks': [
				block.config for block in self.blocks
			],
			'classifier': self.classifier.config,
		}

	@staticmethod
	def build_from_config(config):
		classifier = set_layer_from_config(config['classifier'])

		input_stem = []
		for layer_config in config['input_stem']:
			input_stem.append(set_layer_from_config(layer_config))
		blocks = []
		for block_config in config['blocks']:
			blocks.append(set_layer_from_config(block_config))

		net = ResNets(input_stem, blocks, classifier)
		if 'bn' in config:
			net.set_bn_param(**config['bn'])
		else:
			net.set_bn_param(momentum=0.1, eps=1e-5)

		return net

	def zero_last_gamma(self):
		for m in self.modules():
			if isinstance(m, ResNetBottleneckBlock) and isinstance(m.downsample, IdentityLayer):
				m.conv3.bn.weight.data.zero_()

	@property
	def grouped_block_index(self):
		info_list = []
		block_index_list = []
		for i, block in enumerate(self.blocks):
			if not isinstance(block.downsample, IdentityLayer) and len(block_index_list) > 0:
				info_list.append(block_index_list)
				block_index_list = []
			block_index_list.append(i)
		if len(block_index_list) > 0:
			info_list.append(block_index_list)
		return info_list
	
	def load_state_dict(self, state_dict, **kwargs):
		super(ResNets, self).load_state_dict(state_dict)


