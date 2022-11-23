# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import torch

from model_src.ofa.utils import download_url
from model_src.ofa.imagenet_classification.elastic_nn.networks import OFAMobileNetV3, OFAProxylessNASNets, OFAResNets

__all__ = [ 'ofa_net']



def ofa_net(net_id, pretrained=True, model_dir='.torch/ofa_nets'):
	if net_id == 'ofa_proxyless_d234_e346_k357_w1.3':
		net = OFAProxylessNASNets(
			dropout_rate=0, width_mult=1.3, ks_list=[3, 5, 7], expand_ratio_list=[3, 4, 6], depth_list=[2, 3, 4],
		)
	elif net_id == 'ofa_mbv3_d234_e346_k357_w1.0':
		net = OFAMobileNetV3(
			dropout_rate=0, width_mult=1.0, ks_list=[3, 5, 7], expand_ratio_list=[3, 4, 6], depth_list=[2, 3, 4],
		)
	elif net_id == 'ofa_mbv3_d234_e346_k357_w1.2':
		net = OFAMobileNetV3(
			dropout_rate=0, width_mult=1.2, ks_list=[3, 5, 7], expand_ratio_list=[3, 4, 6], depth_list=[2, 3, 4],
		)
	elif net_id == 'ofa_resnet50':
		net = OFAResNets(
			dropout_rate=0, depth_list=[0, 1, 2], expand_ratio_list=[0.2, 0.25, 0.35], width_mult_list=[0.65, 0.8, 1.0]
		)
		net_id = 'ofa_resnet50_d=0+1+2_e=0.2+0.25+0.35_w=0.65+0.8+1.0'
	else:
		raise ValueError('Not supported: %s' % net_id)

	if pretrained:
		url_base = 'https://hanlab.mit.edu/files/OnceForAll/ofa_nets/'
		init = torch.load(
			download_url(url_base + net_id, model_dir=model_dir),
			map_location='cpu')['state_dict']
		net.load_state_dict(init)
	return net


