import sys
from params import BASE_DIR
# Temp path swap for resolving name conflict on the utils/ folder
# sys.path should not contain too many items, or do not import this file repeatedly
__base_dir_idx = 0
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
else:
    for __di, __dn in enumerate(sys.path):
        if __dn == BASE_DIR:
            __base_dir_idx = __di
            break
    if __base_dir_idx != 0:
        sys.path[0], sys.path[__base_dir_idx] = sys.path[__base_dir_idx], sys.path[0]
if __base_dir_idx != 0:
    sys.path[0], sys.path[__base_dir_idx] = sys.path[__base_dir_idx], sys.path[0]



def _flatten_nested_lists(lists):
    def _recur_flat(val):
        if isinstance(val, list):
            for v in val:
                _recur_flat(v)
        else:
            rv.append(val)
    rv = []
    _recur_flat(lists)
    return rv


def ofa_str_configs_to_subnet_args(net_configs, max_n_net_blocks,
                                   max_n_blocks_per_stage=4,
                                   fill_k=0, fill_e=0,
                                   expected_prefix=None):
    # Input configs must be grouped by stages
    d_list = [len(bs) for bs in net_configs]
    k_list, e_list= [], []
    blocks = iter(_flatten_nested_lists(net_configs))
    for stage_depth in d_list:
        for bi in range(stage_depth):
            block = next(blocks)
            prefix, e_str, k_str = block.split("_")
            if expected_prefix is not None:
                assert expected_prefix == prefix, \
                    "Invalid block prefix: {}, expected: {}".format(prefix, expected_prefix)
            assert e_str.startswith("e"), "Invalid block str: {}".format(block)
            assert k_str.startswith("k"), "Invalid block str: {}".format(block)
            e = int(e_str.replace("e", ""))
            k = int(k_str.replace("k", ""))
            k_list.append(k)
            e_list.append(e)
        for fi in range(max_n_blocks_per_stage - stage_depth):
            # Fill such that k e lists are always the max length
            k_list.append(fill_k)
            e_list.append(fill_e)
    assert len(k_list) >= max_n_net_blocks
    k_list = k_list[:max_n_net_blocks]
    e_list = e_list[:max_n_net_blocks]
    return k_list, e_list, d_list
