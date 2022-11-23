# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.


__all__ = [
	'get_net_device', 
]

""" Network profiling """
def get_net_device(net):
	return net.parameters().__next__().device

