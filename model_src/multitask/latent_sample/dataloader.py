import torch as t
import torchvision.datasets as dset
from torchvision import transforms
import math
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
from model_src.ofa.utils.my_dataloader import MyRandomResizedCrop
import os
from collections import OrderedDict


"""
This is the important file for the project. The dataloader.
"""

# Training and validation transform functions from ImageNet.
# One for each partition.
# https://github.com/mit-han-lab/once-for-all/blob/e9b0e07410e9d9f9e19b2dc0a5a5d66961336cb7/ofa/imagenet_classification/data_providers/imagenet.py#L195
def build_val_transform(size):
    return transforms.Compose([
        transforms.Resize(int(math.ceil(size / 0.875))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def build_train_transform(image_size=224, print_log=False):
    resize_scale = 0.08  # default in OFA
    if print_log:
        print(
            "resize_scale: %s, img_size: %s"
            % (resize_scale, image_size)
        )

    if isinstance(image_size, list):
        resize_transform_class = MyRandomResizedCrop
        print(
            "Use MyRandomResizedCrop: %s, \t %s"
            % MyRandomResizedCrop.get_candidate_image_size(),
            "sync=%s, continuous=%s"
            % (
                MyRandomResizedCrop.SYNC_DISTRIBUTED,
                MyRandomResizedCrop.CONTINUOUS,
            ),
        )
    else:
        resize_transform_class = transforms.RandomResizedCrop

    # random_resize_crop -> random_horizontal_flip
    train_transforms = [
        resize_transform_class(image_size, scale=(resize_scale, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]

    # color augmentation (optional)
    color_transform = transforms.ColorJitter(
        brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
    )
    train_transforms.append(color_transform)

    train_transforms += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    train_transforms = transforms.Compose(train_transforms)
    return train_transforms

class VAECache:
    def __init__(self, latent_levels=["p2", "p3", "p4", "p4_2", "p5"]):
        self.latent_levels = latent_levels
        self._gen_empty_dicts()

    def _gen_empty_dicts(self):
        self.key_dict = {}
        self.tensor_dict = OrderedDict()
        for level in self.latent_levels:
            self.tensor_dict[level] = None
        self.tensor_dict["soft_logits"] = None

    def retrieve(self, index):
        result = []
        tensor_idx = self.key_dict[index]
        for tensor_key in self.tensor_dict.keys():
            result_sublist = self.tensor_dict[tensor_key]
            result.append(result_sublist[0][tensor_idx])
            result.append(result_sublist[1][tensor_idx])
        return result

    def generate_dicts(self, data, processor):
        print("DISCARDING CURRENT DICTS!")
        self._gen_empty_dicts()

        dataloader = t.utils.data.DataLoader(data, batch_size=1, shuffle=False, num_workers=0)

        for index, (image, _, key) in enumerate(dataloader):
            
            key = key[0]
            self.key_dict[key] = index

            latent_list = processor(image)

            for j, level in enumerate(self.tensor_dict.keys()):
                if index == 0:
                    self.tensor_dict[level] = [[latent_list[j][0].cpu()], [latent_list[j][1].cpu()]]
                else:
                    for k, sublist in enumerate(self.tensor_dict[level]):
                        sublist.append(latent_list[j][k].cpu())

        for level in self.tensor_dict.keys():
            new_sublist = []
            for sublist in self.tensor_dict[level]:
                new_sublist.append(t.cat(sublist, dim=0).cpu())
            self.tensor_dict[level] = new_sublist
    
    def save_dicts(self, save_dir):
        import pickle
        os.makedirs(save_dir, exist_ok=True)
        with open(os.sep.join([save_dir, "keys.pkl"]), "wb") as f:
            pickle.dump(self.key_dict, f, protocol=4)

        for tensor_key in self.tensor_dict.keys():
            save_dict = {'mean': self.tensor_dict[tensor_key][0],
                         'sdev': self.tensor_dict[tensor_key][1]}
            t.save(save_dict, os.sep.join([save_dir, "{}.pt".format(tensor_key)]))

    def load_dicts(self, save_dir):
        import pickle
        with open(os.sep.join([save_dir, "keys.pkl"]), "rb") as f:
            self.key_dict = pickle.load(f)
        for tensor_key in self.tensor_dict.keys():
            save_dict = t.load(os.sep.join([save_dir, "{}.pt".format(tensor_key)]))
            tensor_list = [save_dict['mean'], save_dict['sdev']]
            self.tensor_dict[tensor_key] = tensor_list

class VAEDataset(dset.ImageFolder):
    def __init__(self, root, transform, cache = None, **kwargs):
        super(VAEDataset, self).__init__(root=root, transform=transform, **kwargs)
        self.latent_cache = cache

    def __getitem__(self, index):
        key = self._get_key_from_idx(index)
        if self.latent_cache is None:
            return (*super().__getitem__(index), key)
        else:
            return (*super().__getitem__(index), *self.latent_cache.retrieve(key))

    def _get_key_from_idx(self, index):
        idx_str = self.__dict__['samples'][index][0]
        return idx_str[idx_str.rindex(os.sep) + 1:]


class VAEDataloader(t.utils.data.DataLoader):
    def __init__(self, tgt_processor, data, **kwargs):
        super(VAEDataloader, self).__init__(data, **kwargs)
        self.tgt_processor = tgt_processor

    def _get_iterator(self):
        if self.num_workers == 0:
            return SProcessWrapper(self)
        else:
            self.check_worker_number_rationality()
            return MProcessWrapper(self)


class SProcessWrapper(_SingleProcessDataLoaderIter):
    def __init__(self, loader):
        super(SProcessWrapper, self).__init__(loader)
        self.processor = loader.tgt_processor

    def _next_data(self):
        data = super()._next_data()
        latent_tensors = [t for sublist in self.processor(data[0]) for t in sublist]
        return (*data, *latent_tensors)

class MProcessWrapper(_MultiProcessingDataLoaderIter):
    def __init__(self, loader):
        super(MProcessWrapper, self).__init__(loader)
        self.processor = loader.tgt_processor

    def _next_data(self):
        data = super()._next_data()
        latent_tensors = [t for sublist in self.processor(data[0]) for t in sublist]
        return (*data, *latent_tensors)


if __name__ == "__main__":
    from model_src.multitask.latent_sample.arch_sampler import ArchSample
    from model_src.multitask.latent_sample.latent_processor import TargetProcessor

    mySampler = ArchSample("resnet", use_gpu=True)
    myProcessor = TargetProcessor(mySampler, adaptive_pooling=None)
    
    vae_data = VAEDataset(root="/home/.../ImageNet10P/train/", transform=build_train_transform(224, print_log=True))


    # Latent levels for mbv3/pn and ResNet
    #latent_levels=["p2", "p3", "p4", "p4_2", "p5"]
    latent_levels=["p2", "p3", "p4", "p5"]
    myCache = VAECache(latent_levels=latent_levels)
    # Generate dictionaries
    myCache.generate_dicts(vae_data, myProcessor)

    # Once dictionaries are made, save, then wipe and load again, timing each to make sure it all goes off without issue
    import time
    save_start = time.time()
    myCache.save_dicts("./ofa_resnet_trial_dict")
    save_end = time.time()
    print("Time to save: {}".format(save_end - save_start))
    myCache._gen_empty_dicts()
    load_start = time.time()
    myCache.load_dicts("./ofa_resnet_trial_dict")
    load_end = time.time()
    print("Time to load: {}".format(load_end - load_start))