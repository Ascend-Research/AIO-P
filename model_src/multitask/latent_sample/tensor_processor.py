import torch as t
from model_src.multitask.latent_sample.arch_sampler import ArchSample


"""
This script for taking the list of latent representations from arch_sampler.py and compiling them.
"""

class TargetProcessor(t.nn.Module):
    def __init__(self, sampler: ArchSample, adaptive_pooling=None):
        super(TargetProcessor, self).__init__()

        assert adaptive_pooling is None or type(adaptive_pooling) == int
        self.sampler = sampler

        if type(adaptive_pooling) == int:
            self.pooler = t.nn.AdaptiveAvgPool2d((adaptive_pooling, adaptive_pooling))
        else:
            self.pooler = t.nn.Identity()

    def forward(self, x):

        sampled_outputs = self.sampler(x)

        output_list = []
        
        for i, latent_list in enumerate(sampled_outputs):
            latent_tensor = t.cat(latent_list, dim=0)

            (dev_tensor, mean_tensor) = t.std_mean(latent_tensor, dim=0, unbiased=True, keepdim=False)

            if not self.sampler.use_logits or (self.sampler.use_logits and i != len(sampled_outputs) - 1):
                mean_tensor = self.pooler(mean_tensor)
                dev_tensor = self.pooler(dev_tensor)

            output_list.append([mean_tensor, dev_tensor])

        return output_list

