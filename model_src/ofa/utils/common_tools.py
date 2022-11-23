# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import numpy as np
import os
import sys
import torch

try:
	from urllib import urlretrieve
except ImportError:
	from urllib.request import urlretrieve

__all__ = [
	'get_same_padding',
	'sub_filter_start_end', 'min_divisible_value', 'val2list',
	'download_url',
	'accuracy',
	'AverageMeter',
	'DistributedTensor',
]



def get_same_padding(kernel_size):
	if isinstance(kernel_size, tuple):
		assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
		p1 = get_same_padding(kernel_size[0])
		p2 = get_same_padding(kernel_size[1])
		return p1, p2
	assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
	assert kernel_size % 2 > 0, 'kernel size should be odd number'
	return kernel_size // 2


def sub_filter_start_end(kernel_size, sub_kernel_size):
	center = kernel_size // 2
	dev = sub_kernel_size // 2
	start, end = center - dev, center + dev + 1
	assert end - start == sub_kernel_size
	return start, end


def min_divisible_value(n1, v1):
	""" make sure v1 is divisible by n1, otherwise decrease v1 """
	if v1 >= n1:
		return n1
	while n1 % v1 != 0:
		v1 -= 1
	return v1


def val2list(val, repeat_time=1):
	if isinstance(val, list) or isinstance(val, np.ndarray):
		return val
	elif isinstance(val, tuple):
		return list(val)
	else:
		return [val for _ in range(repeat_time)]


def download_url(url, model_dir='~/.torch/', overwrite=False):
	target_dir = url.split('/')[-1]
	model_dir = os.path.expanduser(model_dir)
	try:
		if not os.path.exists(model_dir):
			os.makedirs(model_dir)
		model_dir = os.path.join(model_dir, target_dir)
		cached_file = model_dir
		if not os.path.exists(cached_file) or overwrite:
			sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
			urlretrieve(url, cached_file)
		return cached_file
	except Exception as e:
		# remove lock file so download can be executed next time.
		os.remove(os.path.join(model_dir, 'download.lock'))
		sys.stderr.write('Failed to download from url %s' % url + '\n' + str(e) + '\n')
		return None


def accuracy(output, target, topk=(1,)):
	""" Computes the precision@k for the specified values of k """
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.reshape(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res


class AverageMeter(object):
	"""
	Computes and stores the average and current value
	"""

	def __init__(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class DistributedTensor(object):

	def __init__(self, name):
		self.name = name
		self.sum = None
		self.count = torch.zeros(1)[0]
		self.synced = False

	def update(self, val, delta_n=1):
		val *= delta_n
		if self.sum is None:
			self.sum = val.detach()
		else:
			self.sum += val.detach()
		self.count += delta_n

	@property
	def avg(self):
		import horovod.torch as hvd
		if not self.synced:
			self.sum = hvd.allreduce(self.sum, name=self.name)
			self.synced = True
		return self.sum / self.count
