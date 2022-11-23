import torch as t
import os
from tqdm import tqdm
from params import prepare_global_params
from model_src.multitask.latent_sample.arch_sampler import ArchSample
from model_src.multitask.latent_sample.latent_processor import TargetProcessor
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.config import get_cfg
from detectron2.structures import ImageList
from utils.model_utils import device

class Detectron2Cache:
    def __init__(self, cache_name, latent_levels=["p2", "p3", "p4", "p4_2", "p5"]):
        self.cache_name = cache_name
        self.latent_levels = latent_levels

    def retrieve(self, key):

        latent_list = []
        for i, level in enumerate(self.latent_levels):
            saved_tensor = t.load(os.sep.join([self.cache_name, "{}_{}.pt".format(key, level)]))
            latent_list.append(saved_tensor[0])
            latent_list.append(saved_tensor[1])

        return latent_list
        

    def generate_dicts(self, dataloader, latent_processor):

        for index, img_dict in tqdm(
            enumerate(dataloader),
            # total=len(dataloader),
            desc="Generate dicts",
            ascii=True,
        ):
            key = img_dict["image_id"]
            image = img_dict["image"].unsqueeze(0) # Add batch dimension

            latent_list = latent_processor(image)

            for i, level in enumerate(self.latent_levels):
                mean, sdev = latent_list[i][0], latent_list[i][1]
                tensor_to_save = t.cat([mean, sdev], dim=0)

                t.save(tensor_to_save, os.sep.join([self.cache_name, "{}_{}.pt".format(key, level)]))

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


class Detectron2Dataloader:
    """
    This is the main function that gets the dataloader and preprocessor
        - It will build the detectron2 test or train dataloader
        - It will also do some preprocessing on the image before it is loaded: it will pad the image to the same size as the largest image and/or do sampling
    """

    def __init__(
        self,
        cfg,
        split,
        use_cache=False,
        dataset_name=None,
        latent_levels=["p2", "p3", "p4", "p4_2", "p5"],
    ):
        assert (
            dataset_name is not None if split == "val" else True
        ), "dataset_name is not set when val"

        if split == "train":
            self.dataloader = build_detection_train_loader(cfg)
        else: 
            self.dataloader = build_detection_test_loader(
                cfg, dataset_name=dataset_name
            )
        self.iterator = iter(self.dataloader)

        for attr, value in self.dataloader.__dict__.items():
            self.__dict__[attr] = value
        self.cfg = cfg
        self.split = split
        self.use_cache = use_cache
        self.dataset_name = dataset_name
        self.latent_levels = latent_levels
        self.pixel_mean = t.tensor(self.cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1).to(device())
        self.pixel_std = t.tensor(self.cfg.MODEL.PIXEL_STD).view(-1, 1, 1).to(device())

        self.latent_cache = None
        if use_cache:
            assert (
                cfg.DATALOADER.SAMPLING_CACHE is not None
            ), "cfg.DATALOADER.SAMPLING_CACHE is None"

            self.latent_cache = Detectron2Cache(latent_levels=latent_levels)
            self.latent_cache.load_dicts(
                f"{cfg.DATALOADER.SAMPLING_CACHE}_{split}"
            )

        self.largest_image = t.rand((3, 1333, 1333))

    def __next__(self):
        """
        Before loading the next item, preprocess it!
        """
        next_item = next(self.iterator)[0]
        next_item = self.preprocessor(next_item)
        return next_item

    def preprocessor(self, item):
        """
        Processes the image before it is returned
        - Pads the image to same size as the largest image
        """
        key = item["image_id"]
        item["image"] = self._pad_image(item["image"])

        if self.latent_cache is None:
            return item
        else:
            mean_list, std_dev_list = self.latent_cache.retrieve(key)
            item["mean"] = mean_list
            item["std_dev"] = std_dev_list
            item["input"] = []
            for i in range(len(self.latent_levels)): 
                mean = mean_list[i * 2]
                std_dev = std_dev_list[(i * 2) + 1]
                zeta = t.randn_like(std_dev)
                item["input"].append(mean + zeta * std_dev)
            return item

    def _get_largest_image(self):
        """
        Find the largest image of width, height such that all images can be padded to that same size
        """
        print("Finding largest image in dataset")
        max_size = t.rand((3,1,1)).shape
        for i, img in enumerate(self.dataloader):
            print("i: ",i)
            shape = img[0]["image"].shape
            if shape > max_size:
                max_size = shape

        print("max_size", max_size)
        largest_image = t.rand((3, max_size[-2], max_size[-1]))
        return largest_image

    def _pad_image(self, image):
        image = image.to(device())
        image = (image - self.pixel_mean) / self.pixel_std
        image = ImageList.from_tensors(
            [image, self.largest_image],  # Add largest image in same list
            size_divisibility=32,
        ).tensor
        return image[0]

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.dataloader)


def prepare_local_params(parser):
    parser.add_argument("-n_per_bin", required=False, type=int, default=2)
    parser.add_argument("-config_file", required=True, type=str)
    parser.add_argument("-family", required=True, type=str)
    return parser.parse_args()

def main(params):
    # Prepare config
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.DATALOADER.SAMPLING_CACHE = None
    cfg.set_new_allowed(False)
    cfg.freeze()
    cfg.merge_from_file(params.config_file)

    mySampler = ArchSample(params.family, use_gpu=True, n_per_bin=params.n_per_bin, use_logits=False)
    latent_processor = TargetProcessor(mySampler, adaptive_pooling=None)

    for split in ["train"]:

        cache_name = "cache/new_ofa_{}_detectron2_cache_dict_n{}_{}".format(
            params.family, params.n_per_bin, split
        )

        print(f"Saving to {cache_name}")
        os.makedirs(cache_name, exist_ok=True)
        dataloader = Detectron2Dataloader(
            cfg, split, use_cache=False, dataset_name=cfg.DATASETS.TEST[0]
        )

        latent_levels = ["p2", "p3", "p4", "p4_2", "p5"]
        myCache = Detectron2Cache(cache_name=cache_name, latent_levels=latent_levels)
        myCache.generate_dicts(dataloader, latent_processor)


if __name__ == "__main__":
    _parser = prepare_global_params()
    params = prepare_local_params(_parser)
    main(params)
