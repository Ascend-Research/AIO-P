from constants import *
from params import *
import os
import pickle
from model_src.comp_graph.tf_comp_graph import OP2I
from model_src.comp_graph.tf_comp_graph_utils import compute_cg_flops
from scipy.stats import spearmanr
import numpy as np

"""
Aggregates multiple .pkl cache files into a single
"gpi_{family}_{test_metric}_{suffix}_comp_graph_cache.pkl" file
"""

detectron2_metrics = {
    "obj_det": "obj_det_AP",
    "inst_seg": "inst_seg_AP",
    "sem_seg": "sem_seg_mIoU",
    "pan_seg": "pan_seg_PQ",
}


def prepare_local_params(parser):
    parser.add_argument("-cache_dir", required=True, type=str)
    parser.add_argument('-family', required=True, type=str)
    parser.add_argument("-suffix", required=True, type=str)
    parser.add_argument('-test_metric', required=True, type=str)

    return parser.parse_args()


def main(params):

    cache_contents = os.listdir(params.cache_dir)
    filtered_files = [file for file in cache_contents if file.endswith(".pkl")]
    filtered_files = [file for file in filtered_files if file.startswith(params.family)]

    cg_cache, cg_set = [], set()

    class_accs, task_scores = [], []
    op2i = OP2I().build_from_file()
    for file in filtered_files:
        print("Process file {}".format(file))
        file_path = P_SEP.join([params.cache_dir, file])
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        for cg_dict in data:
            flops_found = False
            cg_name = cg_dict['compute graph'].name
            if cg_name in cg_set:
                print("REPEAT! {}".format(cg_name))
                pass
            else:
                cg_set.add(cg_dict['compute graph'].name)
                class_accs.append(cg_dict['acc'])
                if params.test_metric in detectron2_metrics.keys():
                    task_metric = detectron2_metrics[params.test_metric]
                    cg_dict["acc"] = cg_dict[task_metric] / 100 # Added for decimal format
                    cg_dict["compute graph"] = cg_dict[params.test_metric]
                    task_scores.append(cg_dict["acc"])
                    cg_dict['metric_map'] = "{}->acc".format(task_metric)
                    try:
                        cg_dict['flops'] = cg_dict["flops_{}".format(params.test_metric)]
                        cg_dict['params'] = cg_dict["params_{}".format(params.test_metric)]
                        keys_to_del = list(filter(lambda x: x.startswith("flops_") or x.startswith("params_"), cg_dict.keys()))
                        for k in keys_to_del:
                            del cg_dict[k]
                        flops_found = True
                    except:
                        print("CG {}: count not find task FLOPs/params".format(cg_name))
                else:
                    task_scores.append(cg_dict[params.test_metric])
                    cg_dict['acc'] = task_scores[-1]
                    cg_dict['metric_map'] = "{}->acc".format(params.test_metric)
                    try:
                        cg_dict['flops'] = cg_dict["flops_hpe"]
                        cg_dict['params'] = cg_dict["params_hpe"]
                        del cg_dict['flops_hpe']
                        del cg_dict['params_hpe']
                        flops_found = True
                    except:
                        print("CG {}: could not find task FLOPs/params".format(cg_name))
                cg_dict.pop(params.test_metric, None)
                
                if not flops_found:
                    # Just using the fast counter.
                    print("CG: {}, appending FLOPs using CG counter".format(cg_name))
                    p_flops = compute_cg_flops(cg_dict['compute graph'], op2i, use_fast_counter=True, div=1e9)
                    cg_dict['flops'] = p_flops
                cg_cache.append(cg_dict)

    new_cache_name = "gpi_{}_{}_{}_comp_graph_cache.pkl".format(params.family, params.test_metric, params.suffix)
    new_cache_path = P_SEP.join([CACHE_DIR, new_cache_name])
    with open(new_cache_path, "wb") as f:
        pickle.dump(cg_cache, f, protocol=4)

    rho, p = spearmanr(class_accs, task_scores)
    print("Num Archs total: %d" % len(task_scores))
    print("Compare Classification Accuracy and Test Metric {}".format(params.test_metric))
    print("Classification Acc Dist: %.4f +/- %.4f, [%.4f, %.4f]" % (np.mean(class_accs), np.std(class_accs),
                                                                    np.min(class_accs), np.max(class_accs)))
    print("%s Dist: %.4f +/- %.4f, [%.4f, %.4f]" % (params.test_metric, np.mean(task_scores), np.std(task_scores),
                                                    np.min(task_scores), np.max(task_scores)))

    print("Spearman Correlation = {}; p = {}".format(rho, p))


if __name__ == "__main__":
    _parser = prepare_global_params()
    params = prepare_local_params(_parser)
    main(params)
