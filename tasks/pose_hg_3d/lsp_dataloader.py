import torch as t
import os
from collections import OrderedDict
from tqdm import tqdm
from params import prepare_global_params
from model_src.multitask.latent_sample.arch_sampler import ArchSample
from tasks.pose_hg_3d.lib.datasets.lsp_extended import LSPExtended
from tasks.pose_hg_3d.lib.opts import opts
from model_src.multitask.latent_sample.tensor_processor import TargetProcessor


"""
This is the important file for the project. The dataloader.
"""

class LSPCache:
    def __init__(self, latent_levels=["p5"]):
        self.latent_levels = latent_levels
        self._gen_empty_dicts()

    def _gen_empty_dicts(self):
        self.key_dict = {}
        self.tensor_dict = OrderedDict()
        for level in self.latent_levels:
            self.tensor_dict[level] = None

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

        dataloader = t.utils.data.DataLoader(
            data, batch_size=1, shuffle=False, num_workers=0
        )

        for index, (dict, key) in tqdm(
            enumerate(dataloader),
            total=len(dataloader),
            desc="Generate dicts",
            ascii=True,
        ):
            image = dict["input"]
            self.key_dict[key] = index

            latent_list = processor(image)

            for j, level in enumerate(self.tensor_dict.keys()):
                if index == 0:
                    self.tensor_dict[level] = [
                        [latent_list[j][0].cpu()],
                        [latent_list[j][1].cpu()],
                    ]
                else:
                    for k, sublist in enumerate(self.tensor_dict[level]):
                        sublist.append(latent_list[j][k].cpu())

        for level in self.tensor_dict.keys():
            new_sublist = []
            for sublist in self.tensor_dict[
                level
            ]:  
                new_sublist.append(t.cat(sublist, dim=0).cpu())
            self.tensor_dict[level] = new_sublist

    def save_dicts(self, save_dir):
        import pickle

        os.makedirs(save_dir, exist_ok=True)
        with open(os.sep.join([save_dir, "keys.pkl"]), "wb") as f:
            pickle.dump(self.key_dict, f, protocol=4)

        for tensor_key in self.tensor_dict.keys():
            save_dict = {
                "mean": self.tensor_dict[tensor_key][0],
                "sdev": self.tensor_dict[tensor_key][1],
            }
            t.save(save_dict, os.sep.join([save_dir, "{}.pt".format(tensor_key)]))

    def load_dicts(self, save_dir):
        import pickle

        with open(os.sep.join([save_dir, "keys.pkl"]), "rb") as f:
            self.key_dict = pickle.load(f)
            # [self.key_dict, self.tensor_dict] = pickle.load(f)
        for tensor_key in self.tensor_dict.keys():
            save_dict = t.load(os.sep.join([save_dir, "{}.pt".format(tensor_key)]))
            tensor_list = [save_dict["mean"], save_dict["sdev"]]
            self.tensor_dict[tensor_key] = tensor_list

class LSPDataset(LSPExtended):
    def __init__(self, opt, split, **kwargs):
        super(LSPDataset, self).__init__(opt, split, **kwargs)
        self.latent_cache = None
        if opt.cache_file is not None:
            assert opt.family in opt.cache_file, f"Are you sure you have the correct cache file: {opt.cache_file}?"
            latent_levels = ["p5"]
            self.latent_cache = LSPCache(latent_levels=latent_levels)
            self.latent_cache.load_dicts(f"{opt.cache_file}_{split}") 

    def __getitem__(self, index):
        key = self._get_key_from_idx(index)
        item = super().__getitem__(
            index
        ) 
        if self.latent_cache is None:
            return (item, key)
        else:
            mean, std_dev = self.latent_cache.retrieve(key)
            item["mean"] = mean
            item["std_dev"] = std_dev
            zeta = t.randn_like(std_dev)
            item["input"] = mean + zeta * std_dev
            return item

    def _get_key_from_idx(self, index):
        idx_str = self.__dict__["annot"]["image"][index]
        return idx_str[idx_str.rindex(os.sep) + 1 :]


def prepare_local_params(parser):
    parser.add_argument("-n_per_bin", required=False, type=int, default=2)
    return parser.parse_known_args()


def main(params, unknown_params):
    opt = opts()
    opt = opt.parse(unknown_params)

    mySampler = ArchSample(opt.family, use_gpu=True, n_per_bin=params.n_per_bin, use_logits=False)
    myProcessor = TargetProcessor(mySampler, adaptive_pooling=None)

    for split in ["val", "train"]:
        image_data = LSPDataset(opt, split)
        latent_levels = ["p5"]
        myCache = LSPCache(latent_levels=latent_levels)
        myCache.generate_dicts(image_data, myProcessor)

        cache_name = "cache/ofa_{}_lsp_cache_dict_{}".format(opt.family, split)
        print(f"Saving to {cache_name}")
        myCache.save_dicts(cache_name)


if __name__ == "__main__":
    _parser = prepare_global_params()
    params, unknown_params = prepare_local_params(_parser)
    main(params, unknown_params)
