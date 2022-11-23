import os
import json
import copy
import pickle
import random
import collections
from tqdm import tqdm
from functools import partial
from params import P_SEP, CACHE_DIR, DATA_DIR
from model_src.comp_graph.tf_comp_graph import ComputeGraph, OP2I, load_from_state_dict
from utils.misc_utils import RunningStatMeter, values_to_rank_classes, UniqueList, map_to_range


_DOMAIN_CONFIGS = {
    "classification": { # NOTE: these settings would apply to all families
        "c_in": 3,
        "max_h": 256,
        "max_w": 256,
        "max_kernel_size": 7,
        "max_hidden_size": 512, # May not need to be too large to give normalized values explicitly between 0 and 1
    },
}


_FAMILY2MAX_HW = {
    "ofa": 224,
    "nb101": 32,
    "nb301": 32,
    "nb201c10": 32,
    "nb201c100": 32,
    "nb201imgnet": 32,
}


def get_domain_configs(domain="classification"):
    return _DOMAIN_CONFIGS[domain]


def _build_cache_ofa_pn_mbv3(output_file, log_f=print):
    from model_src.predictor.gpi_data_util import load_gpi_ofa_pn_mbv3_src_data
    from model_src.search_space.ofa_profile.networks_tf import OFAMbv3Net, OFAProxylessNet

    # Make sure duplicate checking is done before this!
    op2idx = OP2I().build_from_file()

    def _model_maker(_configs, _net_func, _name):
        _model = _net_func(_configs, name=_name)
        return lambda _x, training: _model.call(_x, training=training)

    def _single_builder(_configs, _name, _net_func, _h, _w):
        _cg = ComputeGraph(name=_name,
                           H=_h, W=_w,
                           max_kernel_size=_DOMAIN_CONFIGS["classification"]["max_kernel_size"],
                           max_hidden_size=_DOMAIN_CONFIGS["classification"]["max_hidden_size"],
                           max_derived_H=_DOMAIN_CONFIGS["classification"]["max_h"],
                           max_derived_W=_DOMAIN_CONFIGS["classification"]["max_w"])
        _cg.build_from_model_maker(partial(_model_maker, _configs=_configs,
                                           _net_func=_net_func, _name=_name),
                                   op2idx, oov_threshold=0.)
        return _cg

    data = load_gpi_ofa_pn_mbv3_src_data()
    cache_data = []
    bar = tqdm(total=len(data), desc="Building OFA-PN/MBV3 comp graph cache", ascii=True)
    for ni, ((res, net_config), acc, flops, n_params) in enumerate(data):
        net_config_list = copy.deepcopy(net_config)
        if net_config[0][0].startswith("mbconv2"):
            cg = _single_builder(net_config, "OFA-PN-Net{}".format(ni), OFAProxylessNet,
                                 res, res)
        elif net_config[0][0].startswith("mbconv3"):
            cg = _single_builder(net_config, "OFA-MBV3-Net{}".format(ni), OFAMbv3Net,
                                 res, res)
        else:
            raise ValueError("Invalid net configs of OFA: {}".format(net_config))
        cache_data.append({
            "compute graph": cg,
            "acc": acc / 100.,
            "flops": flops,
            "n_params": n_params,
            "original config": (res, net_config_list),
        })
        bar.update(1)
    bar.close()
    random.shuffle(cache_data)
    log_f("Writing {} OFA-PN/MBV3 compute graph data to cache".format(len(cache_data)))
    with open(output_file, "wb") as f:
        pickle.dump(cache_data, f, protocol=4)


def _build_cache_nb301(output_file, log_f=print):
    from model_src.predictor.gpi_data_util import load_gpi_nb301_src_data
    from model_src.darts.model_darts_tf import cifar10_model_maker

    # Make sure duplicate checking is done before this!
    op2idx = OP2I().build_from_file()

    def _single_builder(_geno, _name, _h, _w):
        _cg = ComputeGraph(name=_name,
                           H=_h, W=_w,
                           max_kernel_size=_DOMAIN_CONFIGS["classification"]["max_kernel_size"],
                           max_hidden_size=_DOMAIN_CONFIGS["classification"]["max_hidden_size"],
                           max_derived_H=_DOMAIN_CONFIGS["classification"]["max_h"],
                           max_derived_W=_DOMAIN_CONFIGS["classification"]["max_w"])
        _cg.build_from_model_maker(partial(cifar10_model_maker, genotype=_geno),
                                   op2idx, oov_threshold=0.)
        return _cg

    data = load_gpi_nb301_src_data()
    cache_data = []
    bar = tqdm(total=len(data), desc="Building NB301 comp graph cache", ascii=True)
    for ni, (geno, acc, flops, n_params) in enumerate(data):
        cg = _single_builder(geno, "NB301-Net{}".format(ni), 32, 32)
        cache_data.append({
            "compute graph": cg,
            "acc": acc / 100.,
            "flops": flops,
            "n_params": n_params,
            "original config": geno,
        })
        bar.update(1)
    bar.close()
    random.shuffle(cache_data)
    log_f("Writing {} NB301 compute graph data to cache".format(len(cache_data)))
    with open(output_file, "wb") as f:
        pickle.dump(cache_data, f, protocol=4)


def _build_cache_nb101(output_file, log_f=print):
    from model_src.predictor.gpi_data_util import load_gpi_nb101_src_data
    from model_src.search_space.nb101.example_nb101 import nb101_model_maker

    # Make sure duplicate checking is done before this!
    op2idx = OP2I().build_from_file()

    def _single_builder(_ops, _adj_mat, _name, _h, _w):
        _cg = ComputeGraph(name=_name,
                           H=_h, W=_w,
                           max_kernel_size=_DOMAIN_CONFIGS["classification"]["max_kernel_size"],
                           max_hidden_size=_DOMAIN_CONFIGS["classification"]["max_hidden_size"],
                           max_derived_H=_DOMAIN_CONFIGS["classification"]["max_h"],
                           max_derived_W=_DOMAIN_CONFIGS["classification"]["max_w"])
        _cg.build_from_model_maker(partial(nb101_model_maker, ops=_ops, adj_mat=_adj_mat),
                                   op2idx, oov_threshold=0.)
        return _cg

    data = load_gpi_nb101_src_data()
    cache_data = []
    bar = tqdm(total=len(data), desc="Building NB101 comp graph cache", ascii=True)
    for ni, ((ops, adj_mat), acc, flops, n_params) in enumerate(data):
        cg = _single_builder(ops, adj_mat, "NB101-Net{}".format(ni), 32, 32)
        cache_data.append({
            "compute graph": cg,
            "acc": acc,
            "flops": flops,
            "n_params": n_params,
            "original config": (ops, adj_mat),
        })
        bar.update(1)
    bar.close()
    random.shuffle(cache_data)
    log_f("Writing {} NB101 compute graph data to cache".format(len(cache_data)))
    with open(output_file, "wb") as f:
        pickle.dump(cache_data, f, protocol=4)


def _build_cache_nb201(output_file,
                       src_file=P_SEP.join([CACHE_DIR, "gpi_nb201c10_src_data.pkl"]),
                       n_classes=10, H=32, W=32, log_f=print):
    from model_src.predictor.gpi_data_util import load_gpi_nb201_src_data
    from model_src.search_space.nb201.networks_tf import NB201Net

    log_f("Building NB201 comp graph cache from {}".format(src_file))
    log_f("Number of classes: {}".format(n_classes))
    log_f("H: {}, W: {}".format(H, W))

    op2idx = OP2I().build_from_file()

    def _model_maker(_ops, _input_inds):
        _net = NB201Net(_ops, _input_inds,
                        n_classes=n_classes)
        return _net

    def _single_builder(_ops, _input_inds, _name, _h, _w):
        _cg = ComputeGraph(name=_name,
                           H=_h, W=_w,
                           max_kernel_size=_DOMAIN_CONFIGS["classification"]["max_kernel_size"],
                           max_hidden_size=_DOMAIN_CONFIGS["classification"]["max_hidden_size"],
                           max_derived_H=_DOMAIN_CONFIGS["classification"]["max_h"],
                           max_derived_W=_DOMAIN_CONFIGS["classification"]["max_w"])
        _cg.build_from_model_maker(partial(_model_maker, _ops=_ops, _input_inds=_input_inds),
                                   op2idx, oov_threshold=0.)
        assert len(_cg.nodes) > 10, "Found potentially invalid cg: {}, ops: {}".format(str(_cg), _ops)
        return _cg

    data = load_gpi_nb201_src_data(src_file)
    cache_data = []
    bar = tqdm(total=len(data), desc="Building NB201 comp graph cache", ascii=True)
    for ni, ((ops, op_input_inds), acc, flops) in enumerate(data):
        cg = _single_builder(ops, op_input_inds, "NB201-Net{}".format(ni), H, W)
        cache_data.append({
            "compute graph": cg,
            "acc": acc,
            "flops": flops,
            "original config": (ops, op_input_inds),
        })
        bar.update(1)
    bar.close()
    random.shuffle(cache_data)
    log_f("Writing {} NB201 compute graph data to cache".format(len(cache_data)))
    with open(output_file, "wb") as f:
        pickle.dump(cache_data, f, protocol=4)


def _build_cache_ofa_resnet(output_file, log_f=print):
    from model_src.predictor.gpi_data_util import load_gpi_ofa_resnet_src_data
    from model_src.search_space.ofa_profile.networks_tf import OFAResNet

    op2idx = OP2I().build_from_file()

    def _model_maker(_configs, _net_func, _name):
        _model = _net_func(_configs, name=_name)
        return lambda _x, training: _model.call(_x, training=training)

    def _single_builder(_configs, _name, _net_func, _h, _w):
        _cg = ComputeGraph(name=_name,
                           H=_h, W=_w,
                           max_kernel_size=_DOMAIN_CONFIGS["classification"]["max_kernel_size"],
                           max_hidden_size=_DOMAIN_CONFIGS["classification"]["max_hidden_size"],
                           max_derived_H=_DOMAIN_CONFIGS["classification"]["max_h"],
                           max_derived_W=_DOMAIN_CONFIGS["classification"]["max_w"])
        _cg.build_from_model_maker(partial(_model_maker, _configs=_configs,
                                           _net_func=_net_func, _name=_name),
                                   op2idx, oov_threshold=0.)
        return _cg

    data = load_gpi_ofa_resnet_src_data()
    cache_data = []
    bar = tqdm(total=len(data), desc="Building OFA-ResNet comp graph cache", ascii=True)
    for ni, ((res, net_config), acc, flops, n_params) in enumerate(data):
        cg = _single_builder(copy.deepcopy(net_config),
                             "OFA-ResNet{}".format(ni), OFAResNet,
                             res, res)
        cache_data.append({
            "compute graph": cg,
            "acc": acc / 100.,
            "flops": flops,
            "n_params": n_params,
            "original config": (res, net_config),
        })
        bar.update(1)
    bar.close()
    random.shuffle(cache_data)
    log_f("Writing {} OFA-ResNet compute graph data to cache".format(len(cache_data)))
    with open(output_file, "wb") as f:
        pickle.dump(cache_data, f, protocol=4)


class FamilyDataManager:
    """
    Family-based data manager for the Generalizable Predictor Interface
    Prepares train/dev/test data for each family and combines them
    Also responsible for caching compute graphs
    """
    def __init__(self, families=("nb101", "nb301", "ofa"),
                 family2args=None,
                 cache_dir=CACHE_DIR, data_dir=DATA_DIR,
                 log_f=print):
        self.log_f = log_f
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.families = families
        self.family2args = family2args
        self.validate_cache()

    def get_cache_file_path(self, family):
        return P_SEP.join([self.cache_dir, "gpi_{}_comp_graph_cache.pkl".format(family)])

    def _build_cache(self, family, cache_file):
        if family.lower() == "ofa": 
            _build_cache_ofa_pn_mbv3(cache_file, self.log_f)
        elif family.lower() == "ofa_resnet":
            _build_cache_ofa_resnet(cache_file, self.log_f)
        elif family.lower() == "nb301":
            _build_cache_nb301(cache_file, self.log_f)
        elif family.lower() == "nb101":
            _build_cache_nb101(cache_file, self.log_f)
        elif family.lower() == "nb201c10": # Only 4096 instances
            _build_cache_nb201(cache_file,
                               src_file=P_SEP.join([CACHE_DIR, "gpi_nb201c10_src_data.pkl"]),
                               n_classes=10, H=32, W=32,
                               log_f=self.log_f)
        elif family.lower() == "nb201c100": # Only 4096 instances
            _build_cache_nb201(cache_file, 
                               src_file=P_SEP.join([CACHE_DIR, "gpi_nb201c100_src_data.pkl"]),
                               n_classes=100,H=32, W=32,
                               log_f=self.log_f)
        elif family.lower() == "nb201imgnet": # Only 4096 instances
            _build_cache_nb201(cache_file,
                               src_file=P_SEP.join([CACHE_DIR, "gpi_nb201imgnet_src_data.pkl"]),
                               n_classes=120, H=16, W=16,
                               log_f=self.log_f)
        elif family.lower() == "nb201c10_complete": # Full 15k instances
            _build_cache_nb201(cache_file,
                               src_file=P_SEP.join([CACHE_DIR, "gpi_nb201c10_complete_src_data.pkl"]),
                               n_classes=10, H=32, W=32,
                               log_f=self.log_f)
        elif family.lower() == "nb201c100_complete": # Full 15k instances
            _build_cache_nb201(cache_file,
                               src_file=P_SEP.join([CACHE_DIR, "gpi_nb201c100_complete_src_data.pkl"]),
                               n_classes=100, H=32, W=32,
                               log_f=self.log_f)
        elif family.lower() == "nb201imgnet_complete": # Full 15k instances
            _build_cache_nb201(cache_file,
                               src_file=P_SEP.join([CACHE_DIR, "gpi_nb201imgnet_complete_src_data.pkl"]),
                               n_classes=120, H=16, W=16,
                               log_f=self.log_f)
        else:
            raise ValueError("Unknown family: {}".format(family))

    def validate_cache(self):
        # If compute graph cache is not available for some families, build it
        for f in self.families:

            if f.lower() == "hiaml" or f.lower() == "two_path" or \
                    f.lower() == "inception":
                continue # These families loads from json files

            if f.lower() == "ofa_mbv3" or f.lower() == "ofa_pn":
                f = "ofa" # These are subspaces

            cache_file = self.get_cache_file_path(f)
            if not os.path.isfile(cache_file):
                self.log_f("Building cache for {}".format(f))
                self._build_cache(f, cache_file)

        self.log_f("Cache validated for {}".format(self.families))

    def load_cache_data(self, family):
        if family.lower() == "hiaml" or family.lower() == "two_path":
            # These two families loads from json files
            d = self.get_gpi_custom_set(family_name=family.lower(),
                                        perf_diff_threshold=0,
                                        target_round_n=None,
                                        verbose=False)
            data = [{"compute graph": t[0], "acc": t[1]} for t in d]
        elif family.lower() == "inception":
            d = self.get_inception_custom_set(perf_diff_threshold=0,
                                              target_round_n=None,
                                              verbose=False)
            data = [{"compute graph": t[0], "acc": t[1]} for t in d]
        elif family.lower() == "ofa_mbv3":
            cache_file = self.get_cache_file_path("ofa")
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            subset = []
            for d in data:
                if "mbv3" in d["compute graph"].name.lower():
                    subset.append(d)
            assert len(subset) > 0, "Found empty subset for ofa_mbv3"
            return subset
        elif family.lower() == "ofa_pn":
            cache_file = self.get_cache_file_path("ofa")
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            subset = []
            for d in data:
                if "pn" in d["compute graph"].name.lower():
                    subset.append(d)
            assert len(subset) > 0, "Found empty subset for ofa_pn"
            return subset
        else:
            cache_file = self.get_cache_file_path(family)
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
        return data

    @staticmethod
    def override_cg_max_attrs(data, max_H=None, max_W=None,
                              max_hidden=None, max_kernel=None):
        """
        In-place override of some common cg global attributes
        NOTE: ensure the attribute is not used in any pre-computed features
        """
        for d in data:
            cg = d["compute graph"]
            assert isinstance(cg, ComputeGraph)
            if max_H is not None:
                cg.max_derived_H = max_H
            if max_W is not None:
                cg.max_derived_W = max_W
            if max_hidden is not None:
                cg.max_hidden_size = max_hidden
            if max_kernel is not None:
                cg.max_kernel_size = max_kernel

    def get_src_train_dev_test_sets(self, dev_ratio, test_ratio,
                                    group_by_family=False, shuffle=False,
                                    normalize_HW_per_family=False,
                                    max_hidden_size=None, max_kernel_size=None,
                                    verbose=True):
        # Returns the combined data dicts
        family2data = {}
        for f in self.families:
            if verbose:
                self.log_f("Loading {} cache data...".format(f))
            fd = self.load_cache_data(f)

            if verbose:
                self.log_f("Specified normalize_HW_per_family={}".format(normalize_HW_per_family))
            if normalize_HW_per_family:
                self.override_cg_max_attrs(fd,
                                           max_H=_FAMILY2MAX_HW[f],
                                           max_W=_FAMILY2MAX_HW[f])
            if max_hidden_size is not None:
                if verbose:
                    self.log_f("Override max_hidden_size to {}".format(max_hidden_size))
                self.override_cg_max_attrs(fd, max_hidden=max_hidden_size)
            if max_kernel_size is not None:
                if verbose:
                    self.log_f("Override max_kernel_size to {}".format(max_kernel_size))
                self.override_cg_max_attrs(fd, max_kernel=max_kernel_size)

            if shuffle:
                random.shuffle(fd)

            if self.family2args is not None and \
                "max_size" in self.family2args and \
                f in self.family2args["max_size"]:
                max_size = self.family2args["max_size"][f]
                fd = fd[:max_size]
                if verbose:
                    self.log_f("Specified max total size for {}: {}".format(f, max_size))

            dev_size = max(int(dev_ratio * len(fd)), 1)
            test_size = max(int(test_ratio * len(fd)), 1)
            dev_data = fd[:dev_size]
            test_data = fd[dev_size:dev_size + test_size]
            train_data = fd[dev_size + test_size:]
            if dev_ratio < 1e-5:
                # If dev_ratio == 0, assume there's no dev data and won't care about dev performance
                # Simply merge train/dev data
                train_data += dev_data
                self.log_f("Dev ratio: {} too small, will add dev data to train data".format(dev_ratio))
            if test_ratio < 1e-5:
                # If test_ratio == 0, assume there's no test data and won't care about test performance
                # Simply merge train/test data
                train_data += test_data
                self.log_f("Test ratio: {} too small, will add test data to train data".format(test_ratio))
            family2data[f] = (train_data, dev_data, test_data)
            if verbose:
                self.log_f("Family {} train size: {}".format(f, len(train_data)))
                self.log_f("Family {} dev size: {}".format(f, len(dev_data)))
                self.log_f("Family {} test size: {}".format(f, len(test_data)))
        if group_by_family:
            return family2data
        else:
            train_set, dev_set, test_set = [], [], []
            for f, (train, dev, test) in family2data.items():
                train_set.extend(train)
                dev_set.extend(dev)
                test_set.extend(test)
            random.shuffle(train_set)
            random.shuffle(dev_set)
            random.shuffle(test_set)
            if verbose:
                self.log_f("Combined train size: {}".format(len(train_set)))
                self.log_f("Combined dev size: {}".format(len(dev_set)))
                self.log_f("Combined test size: {}".format(len(test_set)))
            return train_set, dev_set, test_set

    def get_regress_train_dev_test_sets(self, dev_ratio, test_ratio,
                                        normalize_target=False,
                                        normalize_max=None,
                                        group_by_family=False,
                                        normalize_HW_per_family=False,
                                        max_hidden_size=None, max_kernel_size=None,
                                        shuffle=False, perf_key="acc",
                                        verbose=True):

        # Simple function to check for, and return gigaflops
        def check_flops(flops):
            if flops > 1e6:
                return flops / 1e9
            return flops
    
        if group_by_family:
            # Returns a dict of family_str : (train, dev, test)
            family2data = self.get_src_train_dev_test_sets(dev_ratio, test_ratio,
                                                           group_by_family=group_by_family,
                                                           normalize_HW_per_family=normalize_HW_per_family,
                                                           max_hidden_size=max_hidden_size,
                                                           max_kernel_size=max_kernel_size,
                                                           shuffle=shuffle, verbose=verbose)
            tgt_meter = RunningStatMeter()
            rv = {}
            for f, (train_data, dev_data, test_data) in family2data.items():

                fam_tgt_meter = RunningStatMeter()
                train_set, dev_set, test_set = [], [], []
                for d in train_data:
                    train_set.append([d["compute graph"], check_flops(d['flops']), d[perf_key]])
                    tgt_meter.update(d[perf_key])
                    fam_tgt_meter.update(d[perf_key])
                for d in dev_data:
                    dev_set.append([d["compute graph"], check_flops(d['flops']), d[perf_key]])
                    tgt_meter.update(d[perf_key])
                    fam_tgt_meter.update(d[perf_key])
                for d in test_data:
                    test_set.append([d["compute graph"], check_flops(d['flops']), d[perf_key]])
                rv[f] = (train_set, dev_set, test_set)
                if verbose:
                    self.log_f("Max {} target value: {}".format(f, fam_tgt_meter.max))
                    self.log_f("Min {} target value: {}".format(f, fam_tgt_meter.min))
                    self.log_f("Avg {} target value: {}".format(f, fam_tgt_meter.avg))

            if verbose:
                self.log_f("Max global target value: {}".format(tgt_meter.max))
                self.log_f("Min global target value: {}".format(tgt_meter.min))
                self.log_f("Avg global target value: {}".format(tgt_meter.avg))

            if normalize_target:
                if verbose: self.log_f("Normalizing target globally!")
                for _, (train_set, dev_set, test_set) in rv.items():
                    for t in train_set:
                        t[-1] /= tgt_meter.max
                    for t in dev_set:
                        t[-1] /= tgt_meter.max
                    for t in test_set:
                        t[-1] /= tgt_meter.max
                        if normalize_max is not None:
                            t[-1] = min(t[-1], normalize_max)

            return rv

        else:
            # Each instance is (compute graph, perf value)
            train_dicts, dev_dicts, test_dicts = \
                self.get_src_train_dev_test_sets(dev_ratio, test_ratio,
                                                 group_by_family=group_by_family,
                                                 shuffle=shuffle, verbose=verbose)
            tgt_meter = RunningStatMeter()
            train_set, dev_set, test_set = [], [], []
            for d in train_dicts:
                train_set.append([d["compute graph"], check_flops(d['flops']), d[perf_key]])
                tgt_meter.update(d[perf_key])
            for d in dev_dicts:
                dev_set.append([d["compute graph"], check_flops(d['flops']), d[perf_key]])
                tgt_meter.update(d[perf_key])
            for d in test_dicts:
                test_set.append([d["compute graph"], check_flops(d['flops']), d[perf_key]])

            if normalize_target:
                if verbose: self.log_f("Normalizing target globally!")
                for t in train_set:
                    t[-1] /= tgt_meter.max
                for t in dev_set:
                    t[-1] /= tgt_meter.max
                for t in test_set:
                    t[-1] /= tgt_meter.max
                    if normalize_max is not None:
                        t[-1] = min(t[-1], normalize_max)
            if verbose:
                self.log_f("Max target value: {}".format(tgt_meter.max))
                self.log_f("Min target value: {}".format(tgt_meter.min))
                self.log_f("Avg target value: {}".format(tgt_meter.avg))

            return train_set, dev_set, test_set

    def get_irc_train_dev_test_sets(self, dev_ratio, test_ratio, n_classes,
                                    group_by_family=False, perf_key="acc",
                                    normalize_HW_per_family=False,
                                    max_hidden_size=None, max_kernel_size=None,
                                    shuffle=False, verbose=True):
        # Independent rank classification on custom percentile intervals
        # Each instance is (compute graph, rank class)
        # Pre-splitting of train dev test sets may lead to an unevenly class distribution
        # For now we do not do anything about this, just reporting it
        # Higher class number = higher percentile rank = better
        family2sets = \
            self.get_src_train_dev_test_sets(dev_ratio, test_ratio,
                                             group_by_family=True,
                                             normalize_HW_per_family=normalize_HW_per_family,
                                             max_hidden_size=max_hidden_size,
                                             max_kernel_size=max_kernel_size,
                                             shuffle=shuffle, verbose=verbose)
        family2data = {}
        for f, (train, dev, test) in family2sets.items():
            _, thresholds = values_to_rank_classes(train + dev, n_classes,
                                                   value_getter=lambda _d: _d[perf_key])
            train_classes, _ = values_to_rank_classes(train, n_classes,
                                                      value_getter=lambda _d: _d[perf_key],
                                                      thresholds=thresholds)
            dev_classes, _ = values_to_rank_classes(dev, n_classes,
                                                    value_getter=lambda _d: _d[perf_key],
                                                    thresholds=thresholds)
            test_classes, _ = values_to_rank_classes(test, n_classes,
                                                     value_getter=lambda _d: _d[perf_key],
                                                     thresholds=thresholds)

            train_set, dev_set, test_set = [], [], []
            for d, c in zip(train, train_classes):
                train_set.append((d["compute graph"], c))
            for d, c in zip(dev, dev_classes):
                dev_set.append((d["compute graph"], c))
            for d, c in zip(test, test_classes):
                test_set.append((d["compute graph"], c))
            family2data[f] = (train_set, dev_set, test_set)

            if verbose:
                class2count_train = collections.defaultdict(int)
                class2count_dev = collections.defaultdict(int)
                class2count_test = collections.defaultdict(int)
                for c in train_classes:
                    class2count_train[c] += 1
                for c in dev_classes:
                    class2count_dev[c] += 1
                for c in test_classes:
                    class2count_test[c] += 1
                self.log_f("Rank class threshold for family {}: {}".format(f, thresholds))
                self.log_f("Train class distribution: {}".format(
                    sorted([(k, v) for k, v in class2count_train.items()], key=lambda t:t[0])))
                self.log_f("Dev class distribution: {}".format(
                    sorted([(k, v) for k, v in class2count_dev.items()], key=lambda t:t[0])))
                self.log_f("Test class distribution: {}".format(
                    sorted([(k, v) for k, v in class2count_test.items()], key=lambda t:t[0])))
                self.log_f("")

        if group_by_family:
            return family2data
        else:
            global_train, global_dev, global_test = [], [], []
            for _, (fam_train, fam_dev, fam_test) in family2data.items():
                global_train.extend(fam_train)
                global_dev.extend(fam_dev)
                global_test.extend(fam_test)
            random.shuffle(global_train)
            random.shuffle(global_dev)
            random.shuffle(global_test)
            return global_train, global_dev, global_test

    def get_fc_train_dev_test_sets(self, dev_ratio, test_ratio,
                                   normalize_HW_per_family=False,
                                   max_hidden_size=None, max_kernel_size=None,
                                   shuffle=False, verbose=True):
        # Family classification
        # Each instance is (compute graph, family class)
        family2sets = \
            self.get_src_train_dev_test_sets(dev_ratio, test_ratio,
                                             group_by_family=True,
                                             normalize_HW_per_family=normalize_HW_per_family,
                                             max_hidden_size=max_hidden_size,
                                             max_kernel_size=max_kernel_size,
                                             shuffle=shuffle, verbose=verbose)

        train_set, dev_set, test_set = [], [], []
        family_idx = 0
        idx2family = {}
        for f, (train, dev, test) in family2sets.items():
            idx2family[family_idx] = f
            for d in train:
                train_set.append((d["compute graph"], family_idx))
            for d in dev:
                dev_set.append((d["compute graph"], family_idx))
            for d in test:
                test_set.append((d["compute graph"], family_idx))
            family_idx += 1
        if verbose:
            self.log_f("Idx2family: {}".format(idx2family))
        return train_set, dev_set, test_set

    def get_ipc_train_dev_test_sets(self, dev_ratio, test_ratio,
                                    normalize_HW_per_family=False,
                                    max_hidden_size=None, max_kernel_size=None,
                                    group_by_family=False,
                                    symmetric_pairs=False,
                                    max_n_train=None,
                                    max_n_dev=None,
                                    max_n_test=None,
                                    tri_state=False,
                                    perf_key="acc",
                                    shuffle=False,
                                    delta=1e-4,
                                    verbose=True):
        # Independent pairwise classification
        # Each instance is (compute graph 1, compute graph 2, comparison class)
        family2sets = \
            self.get_src_train_dev_test_sets(dev_ratio, test_ratio,
                                             group_by_family=True,
                                             normalize_HW_per_family=normalize_HW_per_family,
                                             max_hidden_size=max_hidden_size,
                                             max_kernel_size=max_kernel_size,
                                             shuffle=shuffle, verbose=verbose)

        if verbose:
            self.log_f("IPC delta: {}".format(delta))
            self.log_f("IPC tri-state: {}".format(tri_state))
            self.log_f("IPC symmetric_pairs: {}".format(symmetric_pairs))

        def _build_pairs(data, unique_pairs, desc,
                         max_n_instances=None):
            size = int(len(data) ** 2)
            if max_n_instances is not None:
                size = min(size, max_n_instances)
            if symmetric_pairs:
                size *= 2
            bar = None
            if verbose:
                bar = tqdm(total=size, desc=desc, ascii=True)
            pairs = []
            sorted_data = sorted(data, key=lambda d: d[perf_key])
            sample_larger = False
            while len(pairs) < size:
                # Stratified sampling
                idx1 = random.choice(list(range(len(sorted_data))))
                if idx1 == 0 and not sample_larger:
                    continue
                elif idx1 == len(sorted_data) - 1 and sample_larger:
                    continue
                if sample_larger:
                    start, end = idx1 + 1, len(sorted_data)
                else:
                    start, end = 0, idx1
                idx2 = random.choice(list(range(start, end)))
                d1, d2 = sorted_data[idx1], sorted_data[idx2]

                # Class building
                d1_perf = d1[perf_key]
                d2_perf = d2[perf_key]
                if abs(d1_perf - d2_perf) < delta and not tri_state:
                    continue
                if abs(d1_perf - d2_perf) < delta:
                    cls, inv_cls = 2, 2
                else:
                    cls = 0 if d1_perf < d2_perf else 1
                    inv_cls = 1 - cls

                # Pair collection
                # Use str() as id works here because each compute graph's name contains its index
                pair_id = (str(d1["compute graph"]), str(d2["compute graph"]))
                if pair_id not in unique_pairs:
                    unique_pairs.add(pair_id)
                    pairs.append([d1["compute graph"], d2["compute graph"], cls])
                    sample_larger = not sample_larger
                    if bar is not None:
                        bar.update(1)
                    if symmetric_pairs:
                        pairs.append([d2["compute graph"], d1["compute graph"], inv_cls])
                        if bar is not None:
                            bar.update(1)

            if bar is not None: bar.close()
            return pairs

        family2data = {}
        unique_pairs_set = set()
        for f, (train, dev, test) in family2sets.items():
            train_pairs = _build_pairs(train, unique_pairs_set,
                                       desc="Building {} train pairs".format(f),
                                       max_n_instances=max_n_train)
            dev_pairs = _build_pairs(dev, unique_pairs_set,
                                     desc="Building {} dev pairs".format(f),
                                     max_n_instances=max_n_dev)
            test_pairs = _build_pairs(test, unique_pairs_set,
                                      desc="Building {} test pairs".format(f),
                                      max_n_instances=max_n_test)
            family2data[f] = (train_pairs, dev_pairs, test_pairs)

            if verbose:
                class2count_train = collections.defaultdict(int)
                class2count_dev = collections.defaultdict(int)
                class2count_test = collections.defaultdict(int)
                for _, _, c in train_pairs:
                    class2count_train[c] += 1
                for _, _, c in dev_pairs:
                    class2count_dev[c] += 1
                for _, _, c in test_pairs:
                    class2count_test[c] += 1
                self.log_f("Train class distribution: {}".format(
                    sorted([(k, v) for k, v in class2count_train.items()], key=lambda t: t[0])))
                self.log_f("Dev class distribution: {}".format(
                    sorted([(k, v) for k, v in class2count_dev.items()], key=lambda t: t[0])))
                self.log_f("Test class distribution: {}".format(
                    sorted([(k, v) for k, v in class2count_test.items()], key=lambda t: t[0])))
                self.log_f("")

        if group_by_family:
            return family2data
        else:
            global_train, global_dev, global_test = [], [], []
            for _, (fam_train, fam_dev, fam_test) in family2data.items():
                global_train.extend(fam_train)
                global_dev.extend(fam_dev)
                global_test.extend(fam_test)
            random.shuffle(global_train)
            random.shuffle(global_dev)
            random.shuffle(global_test)
            return global_train, global_dev, global_test

    def get_igr_train_dev_test_sets(self, dev_ratio, test_ratio,
                                    normalize_HW_per_family=False,
                                    max_hidden_size=None, max_kernel_size=None,
                                    round_n=3, re_map_range=True,
                                    fam_perf_threshold=None,
                                    common_range_max=10.0,
                                    common_range_min=1.0,
                                    shuffle=False,
                                    perf_key="acc",
                                    verbose=True):
        # Independent group regression
        # Round each perf value, then project to different ranges
        # Directly regress the bucket value for an arch
        # Truth data collected on a per-family basis
        family2sets = \
            self.get_src_train_dev_test_sets(dev_ratio, test_ratio,
                                             group_by_family=True,
                                             normalize_HW_per_family=normalize_HW_per_family,
                                             max_hidden_size=max_hidden_size,
                                             max_kernel_size=max_kernel_size,
                                             shuffle=shuffle, verbose=verbose)

        if verbose:
            self.log_f("IGR round n: {}".format(round_n))
            self.log_f("IGR re-map: {}".format(re_map_range))

        family2data = self.gen_igr_family2data(family2sets, common_range_min, common_range_max,
                                               verbose=verbose, re_map_range=re_map_range,
                                               fam_perf_threshold=fam_perf_threshold,
                                               perf_key=perf_key, round_n=round_n)

        return family2data

    def gen_igr_family2data(self, family2sets,
                            common_range_min,
                            common_range_max,
                            verbose=True,
                            re_map_range=True,
                            fam_perf_threshold=None,
                            perf_key="acc", round_n=3):
        # Returns a family2data dict instead of merging everything together
        family2data = {}
        for f, (train, dev, test) in family2sets.items():
            train_pairs, dev_pairs, test_pairs = [], [], []

            # First collect the bucket and boundary values for family
            family_tgt_meter = RunningStatMeter()
            buckets = UniqueList()
            for d in train + dev:
                perf = round(d[perf_key], round_n)

                if fam_perf_threshold is not None and \
                        f in fam_perf_threshold and \
                        perf < fam_perf_threshold[f]:
                    continue

                buckets.append(perf)
                family_tgt_meter.update(perf)
            buckets = buckets.tolist()
            buckets.sort(reverse=True)
            rv_buckets = []
            for p in buckets:
                if re_map_range:
                    label = map_to_range(p, family_tgt_meter.min, family_tgt_meter.max,
                                         common_range_min, common_range_max)
                else:
                    label = p
                rv_buckets.append((label, p))

            if verbose:
                self.log_f("Collected {} rounded buckets for family {}".format(len(rv_buckets), f))
                self.log_f("Top-5 bucket values: {}".format(rv_buckets[:5]))
                self.log_f("Bottom-5 bucket values: {}".format(rv_buckets[::-1][:5]))

            # Then collect data instances and map them to new range
            # If no mapping is specified, we are essentially predicting the rounded perfs
            final_tgt_meter = RunningStatMeter()
            for d in train:
                perf = round(d[perf_key], round_n)

                if fam_perf_threshold is not None and \
                        f in fam_perf_threshold and \
                        perf < fam_perf_threshold[f]:
                    continue

                if re_map_range:
                    perf = map_to_range(perf, family_tgt_meter.min, family_tgt_meter.max,
                                        common_range_min, common_range_max)
                final_tgt_meter.update(perf)
                train_pairs.append([d["compute graph"], perf])
            for d in dev:
                perf = round(d[perf_key], round_n)

                if fam_perf_threshold is not None and \
                        f in fam_perf_threshold and \
                        perf < fam_perf_threshold[f]:
                    continue

                if re_map_range:
                    perf = map_to_range(perf, family_tgt_meter.min, family_tgt_meter.max,
                                        common_range_min, common_range_max)
                final_tgt_meter.update(perf)
                dev_pairs.append([d["compute graph"], perf])
            for d in test:
                perf = round(d[perf_key], round_n)

                if fam_perf_threshold is not None and \
                        f in fam_perf_threshold and \
                        perf < fam_perf_threshold[f]:
                    continue

                if re_map_range:
                    perf = map_to_range(perf, family_tgt_meter.min, family_tgt_meter.max,
                                        common_range_min, common_range_max)
                final_tgt_meter.update(perf)
                test_pairs.append([d["compute graph"], perf])

            if verbose:
                self.log_f("Max final target value for {}: {}".format(f, final_tgt_meter.max))
                self.log_f("Min final target value for {}: {}".format(f, final_tgt_meter.min))
                self.log_f("Avg final target value for {}: {}".format(f, final_tgt_meter.avg))
                self.log_f("")

            family2data[f] = (train_pairs, dev_pairs, test_pairs, rv_buckets)

        return family2data

    def get_idr_train_dev_test_sets(self, dev_ratio, test_ratio,
                                    normalize_HW_per_family=False,
                                    max_hidden_size=None, max_kernel_size=None,
                                    max_abs_diff_value=0.1,
                                    perf_multiplier=1.,
                                    group_by_family=False,
                                    symmetric_pairs=False,
                                    max_n_train=None,
                                    max_n_dev=None,
                                    max_n_test=None,
                                    perf_key="acc",
                                    shuffle=False,
                                    delta=1e-4,
                                    verbose=True):
        # Independent difference regression
        # Each instance is (compute graph 1, compute graph 2, diff in perf)
        family2sets = \
            self.get_src_train_dev_test_sets(dev_ratio, test_ratio,
                                             group_by_family=True,
                                             normalize_HW_per_family=normalize_HW_per_family,
                                             max_hidden_size=max_hidden_size,
                                             max_kernel_size=max_kernel_size,
                                             shuffle=shuffle, verbose=verbose)

        if verbose:
            self.log_f("IDR delta: {}".format(delta))
            self.log_f("IDR symmetric_pairs: {}".format(symmetric_pairs))
            self.log_f("IDR max_abs_diff_value: {}".format(max_abs_diff_value))

        def _build_pairs(data, unique_pairs, desc,
                         max_n_instances=None):
            size = int(len(data) ** 2)
            if max_n_instances is not None:
                size = min(size, max_n_instances)
            if symmetric_pairs:
                size *= 2
            bar = None
            if verbose:
                bar = tqdm(total=size, desc=desc, ascii=True)
            pairs = []
            sorted_data = sorted(data, key=lambda d: d[perf_key])
            sample_larger = False
            while len(pairs) < size:
                # Stratified sampling
                idx1 = random.choice(list(range(len(sorted_data))))
                if idx1 == 0 and not sample_larger:
                    continue
                elif idx1 == len(sorted_data) - 1 and sample_larger:
                    continue
                if sample_larger:
                    start, end = idx1 + 1, len(sorted_data)
                else:
                    start, end = 0, idx1
                idx2 = random.choice(list(range(start, end)))
                d1, d2 = sorted_data[idx1], sorted_data[idx2]

                # Target building
                d1_perf = d1[perf_key] * perf_multiplier
                d2_perf = d2[perf_key] * perf_multiplier
                if abs(d1_perf - d2_perf) < delta:
                    continue
                tgt = d1_perf - d2_perf
                if abs(tgt) > max_abs_diff_value:
                    if tgt > 0:
                        tgt = max_abs_diff_value
                    else:
                        tgt = -max_abs_diff_value

                # Pair collection
                pair_id = (str(d1["compute graph"]), str(d2["compute graph"]))
                if pair_id not in unique_pairs:
                    unique_pairs.add(pair_id)
                    pairs.append([d1["compute graph"], d2["compute graph"], tgt])
                    sample_larger = not sample_larger
                    if bar is not None:
                        bar.update(1)
                    if symmetric_pairs:
                        pairs.append([d2["compute graph"], d1["compute graph"], -tgt])
                        if bar is not None:
                            bar.update(1)

            if bar is not None: bar.close()
            return pairs

        family2data = {}
        unique_pairs_set = set()
        for f, (train, dev, test) in family2sets.items():
            train_pairs = _build_pairs(train, unique_pairs_set,
                                       desc="Building {} train pairs".format(f),
                                       max_n_instances=max_n_train)
            dev_pairs = _build_pairs(dev, unique_pairs_set,
                                     desc="Building {} dev pairs".format(f),
                                     max_n_instances=max_n_dev)
            test_pairs = _build_pairs(test, unique_pairs_set,
                                      desc="Building {} test pairs".format(f),
                                      max_n_instances=max_n_test)
            family2data[f] = (train_pairs, dev_pairs, test_pairs)

            if verbose:
                family_tgt_meter = RunningStatMeter()
                for _, _, t in train_pairs:
                    family_tgt_meter.update(t)
                for _, _, t in dev_pairs:
                    family_tgt_meter.update(t)
                for _, _, t in test_pairs:
                    family_tgt_meter.update(t)
                self.log_f("Max final target value for {}: {}".format(f, family_tgt_meter.max))
                self.log_f("Min final target value for {}: {}".format(f, family_tgt_meter.min))
                self.log_f("Avg final target value for {}: {}".format(f, family_tgt_meter.avg))
                self.log_f("")

        if group_by_family:
            return family2data
        else:
            global_train, global_dev, global_test = [], [], []
            for _, (fam_train, fam_dev, fam_test) in family2data.items():
                global_train.extend(fam_train)
                global_dev.extend(fam_dev)
                global_test.extend(fam_test)
            random.shuffle(global_train)
            random.shuffle(global_dev)
            random.shuffle(global_test)
            return global_train, global_dev, global_test

    @staticmethod
    def get_efficient_net_test_set(normalize_HW_per_family=False,
                                   max_hidden_size=None, max_kernel_size=None):
        from model_src.efficientnet.model_efficientnet_tf import simple_model_maker
        data = []
        acc_list = [77.1, 79.1, 80.1, 81.6, 82.9, 83.6, 84.0, 84.3]
        op2idx = OP2I().build_from_file()
        for i in range(8):
            model_name = "efficientnet-b{}".format(i)
            model_maker = partial(simple_model_maker, model_name=model_name, training=False)
            if max_hidden_size is None:
                max_hidden_size = _DOMAIN_CONFIGS["classification"]["max_hidden_size"]
            if max_kernel_size is None:
                max_kernel_size = _DOMAIN_CONFIGS["classification"]["max_kernel_size"]
            if normalize_HW_per_family:
                max_derived_H, max_derived_W = 224, 224
            else:
                max_derived_H = _DOMAIN_CONFIGS["classification"]["max_h"]
                max_derived_W = _DOMAIN_CONFIGS["classification"]["max_w"]
            cg = ComputeGraph(name="EfficientNet-B{}".format(i),
                              max_kernel_size=max_kernel_size,
                              max_hidden_size=max_hidden_size,
                              max_derived_H=max_derived_H,
                              max_derived_W=max_derived_W)
            cg.build_from_model_maker(model_maker, op2idx,
                                      oov_threshold=0.)
            data.append([cg, acc_list[i] / 100.])
        return data

    def get_nb201_test_set(self, family_name="nb201c10",
                           n_nets=None, ordered=False,
                           normalize_HW_per_family=False,
                           max_hidden_size=None, max_kernel_size=None,
                           perf_diff_threshold=2e-4,
                           perf_key="acc", verbose=True):
        if verbose:
            self.log_f("Loading {} cache data...".format(family_name))

        fd = self.load_cache_data(family_name)

        if verbose:
            self.log_f("Specified normalize_HW_per_family={}".format(normalize_HW_per_family))
        if normalize_HW_per_family:
            self.override_cg_max_attrs(fd,
                                       max_H=_FAMILY2MAX_HW[family_name],
                                       max_W=_FAMILY2MAX_HW[family_name])
        if max_hidden_size is not None:
            if verbose:
                self.log_f("Override max_hidden_size to {}".format(max_hidden_size))
            self.override_cg_max_attrs(fd, max_hidden=max_hidden_size)
        if max_kernel_size is not None:
            if verbose:
                self.log_f("Override max_kernel_size to {}".format(max_kernel_size))
            self.override_cg_max_attrs(fd, max_kernel=max_kernel_size)

        if ordered:
            if verbose: self.log_f("Specified ordered={}".format(ordered))
            fd.sort(key=lambda _d:_d[perf_key], reverse=True)

        if n_nets is not None:
            if verbose: self.log_f("Specified num nets: {}".format(n_nets))
            fd = fd[:n_nets]

        rv = [[_d["compute graph"], _d[perf_key]] for _d in fd]

        if perf_diff_threshold is not None:
            if verbose: self.log_f("Specified perf diff threshold={}".format(perf_diff_threshold))
            sorted_data = sorted(rv, key=lambda _t: _t[1], reverse=True)
            pruned_indices = set()
            for i, (g, p) in enumerate(sorted_data):
                prev_idx = i - 1
                while prev_idx > 0 and prev_idx in pruned_indices:
                    prev_idx -= 1
                if prev_idx >= 0 and abs(p - sorted_data[prev_idx][1]) < perf_diff_threshold:
                    pruned_indices.add(i)
            rv = [_t for i, _t in enumerate(sorted_data) if i not in pruned_indices]

        if verbose:
            tgt_meter = RunningStatMeter()
            for _, t in rv:
                tgt_meter.update(t)
            self.log_f("Max final target value for {}: {}".format(family_name, tgt_meter.max))
            self.log_f("Min final target value for {}: {}".format(family_name, tgt_meter.min))
            self.log_f("Avg final target value for {}: {}".format(family_name, tgt_meter.avg))
            self.log_f("Loaded {} {} instances".format(len(rv), family_name))

        return rv

    def get_gpi_custom_set(self, family_name="hiaml", dataset="cifar10",
                           max_hidden_size=None, max_kernel_size=None,
                           perf_diff_threshold=2e-4, target_round_n=None,
                           verbose=True):
        if verbose:
            self.log_f("Loading {} data...".format(family_name))

        data_file = P_SEP.join([self.data_dir, "gpi_test_{}_{}_labelled_cg_data.json".format(family_name, dataset)])

        with open(data_file, "r") as f:
            data = json.load(f)

        if max_hidden_size is None:
            max_hidden_size = _DOMAIN_CONFIGS["classification"]["max_hidden_size"]
        if max_kernel_size is None:
            max_kernel_size = _DOMAIN_CONFIGS["classification"]["max_kernel_size"]
        max_derived_H = _DOMAIN_CONFIGS["classification"]["max_h"]
        max_derived_W = _DOMAIN_CONFIGS["classification"]["max_w"]

        rv = []
        bar = None
        if verbose:
            bar = tqdm(total=len(data), desc="Inflating compute graphs", ascii=True)
        for k, v in data.items():
            cg = ComputeGraph(name="", H=32, W=32, C_in=3,
                              max_hidden_size=max_hidden_size, max_kernel_size=max_kernel_size,
                              max_derived_H=max_derived_H, max_derived_W=max_derived_W)
            cg = load_from_state_dict(cg, v["cg"])
            acc = v["max_perf"] / 100.
            rv.append((cg, acc))
            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()

        if perf_diff_threshold is not None:
            if verbose: self.log_f("Specified perf diff threshold={}".format(perf_diff_threshold))
            sorted_data = sorted(rv, key=lambda _t: _t[1], reverse=True)
            pruned_indices = set()
            for i, (g, p) in enumerate(sorted_data):
                prev_idx = i - 1
                while prev_idx > 0 and prev_idx in pruned_indices:
                    prev_idx -= 1
                if prev_idx >= 0 and abs(p - sorted_data[prev_idx][1]) < perf_diff_threshold:
                    pruned_indices.add(i)
            rv = [_t for i, _t in enumerate(sorted_data) if i not in pruned_indices]

        if target_round_n is not None:
            if verbose: self.log_f("Specified target round n={}".format(target_round_n))
            rv = [(c, round(t, target_round_n)) for c, t in rv]

        if verbose:
            tgt_meter = RunningStatMeter()
            for _, t in rv:
                tgt_meter.update(t)
            self.log_f("Max final target value for {}: {}".format(family_name, tgt_meter.max))
            self.log_f("Min final target value for {}: {}".format(family_name, tgt_meter.min))
            self.log_f("Avg final target value for {}: {}".format(family_name, tgt_meter.avg))
            self.log_f("Loaded {} {} instances".format(len(rv), family_name))

        return rv

    def get_inception_custom_set(self, dataset="cifar10",
                                 max_hidden_size=None, max_kernel_size=None,
                                 perf_diff_threshold=None, target_round_n=None,
                                 verbose=True):

        data_file = P_SEP.join([self.data_dir, "inception_{}_labelled_cg_data.json".format(dataset)])

        with open(data_file, "r") as f:
            data = json.load(f)

        if max_hidden_size is None:
            max_hidden_size = _DOMAIN_CONFIGS["classification"]["max_hidden_size"]
        if max_kernel_size is None:
            max_kernel_size = _DOMAIN_CONFIGS["classification"]["max_kernel_size"]
        max_derived_H = _DOMAIN_CONFIGS["classification"]["max_h"]
        max_derived_W = _DOMAIN_CONFIGS["classification"]["max_w"]

        rv = []
        bar = None
        if verbose:
            bar = tqdm(total=len(data), desc="Inflating compute graphs", ascii=True)
        for k, v in data.items():
            cg = ComputeGraph(name="", H=32, W=32, C_in=3,
                              max_hidden_size=max_hidden_size, max_kernel_size=max_kernel_size,
                              max_derived_H=max_derived_H, max_derived_W=max_derived_W)
            cg = load_from_state_dict(cg, v["cg"])
            acc = v["max_perf"] / 100.
            rv.append((cg, acc))
            if bar is not None:
                bar.update(1)
        if bar is not None:
            bar.close()

        if perf_diff_threshold is not None:
            if verbose: self.log_f("Specified perf diff threshold={}".format(perf_diff_threshold))
            sorted_data = sorted(rv, key=lambda _t: _t[1], reverse=True)
            pruned_indices = set()
            for i, (g, p) in enumerate(sorted_data):
                prev_idx = i - 1
                while prev_idx > 0 and prev_idx in pruned_indices:
                    prev_idx -= 1
                if prev_idx >= 0 and abs(p - sorted_data[prev_idx][1]) < perf_diff_threshold:
                    pruned_indices.add(i)
            rv = [_t for i, _t in enumerate(sorted_data) if i not in pruned_indices]

        if target_round_n is not None:
            if verbose: self.log_f("Specified target round n={}".format(target_round_n))
            rv = [(c, round(t, target_round_n)) for c, t in rv]

        if verbose:
            tgt_meter = RunningStatMeter()
            for _, t in rv:
                tgt_meter.update(t)
            self.log_f("Max final target value for inception: {}".format(tgt_meter.max))
            self.log_f("Min final target value for inception: {}".format(tgt_meter.min))
            self.log_f("Avg final target value for inception: {}".format(tgt_meter.avg))
            self.log_f("Loaded {} inception instances".format(len(rv)))

        return rv
