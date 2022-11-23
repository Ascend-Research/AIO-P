import random
import numpy as np
import collections
from tqdm import tqdm
from utils.math_utils import mean, variance
from sklearn.metrics import mean_squared_error, mean_absolute_error


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def get_regression_metrics(pred_list, target_list):
    assert len(pred_list) == len(target_list), \
        "pred len: {}, target len: {}".format(len(pred_list), len(target_list))
    n = len(pred_list)
    mean_sq_error = mean_squared_error(target_list, pred_list)
    mean_abs_error = mean_absolute_error(target_list, pred_list)
    max_err = max([abs(target_list[i] - pred_list[i]) for i in range(n)])
    if any(abs(v) < 1e-9 for v in target_list):
        mape_pred_list, mape_target_list = [], []
        for i, v in enumerate(target_list):
            if abs(v) < 1e-9: continue
            mape_pred_list.append(pred_list[i])
            mape_target_list.append(v)
    else:
        mape_pred_list, mape_target_list = pred_list, target_list
    mape = 100 * mean([abs(mape_pred_list[i] - truth) / abs(truth) for i, truth in enumerate(mape_target_list)],
                      fallback_val=0)
    pred_mean = mean(pred_list)
    pred_variance = variance(pred_list)
    truth_mean = mean(target_list)
    truth_variance = variance(target_list)
    rv = {
        "mean_square_error": mean_sq_error,
        "mean_absolute_error": mean_abs_error,
        "max_error": max_err,
        "mean_absolute_percent_error": mape,
        "mape_effective_sample_size_diff": abs(len(mape_pred_list) - len(pred_list)),
        "pred_mean": pred_mean,
        "pred_variance": pred_variance,
        "truth_mean": truth_mean,
        "truth_variance": truth_variance,
    }
    return rv



def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)

    Relevance is positive real values.  Can use binary
    as the previous methods.

    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114

    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]

    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def get_regression_rank_metrics(pred_list, truth_list, ndcg_p_list=(),
                                verbose=False, top_overlap_k_list=(5, 10, 50, 100)):

    def _compute_global_ranking_metrics():
        from scipy.stats import spearmanr
        spearman_rho, spearman_p = spearmanr(pred_list, truth_list)
        metrics["spearman_rho"] = spearman_rho
        metrics["spearman_p"] = spearman_p

    def _compute_multi_cand_ranking_metrics(curr_i, key_prefix, _cand_indices, k_cands):
        ranked_pred_vals = [(ci, pred_target_list[ci]) for ci in _cand_indices]
        ranked_pred_vals.append((curr_i, pred_val))
        ranked_pred_vals.sort(key=lambda t: t[1], reverse=True)
        ranked_truth_vals = [(ci, truth_target_list[ci]) for ci in _cand_indices]
        ranked_truth_vals.append((curr_i, pred_val))
        ranked_truth_vals.sort(key=lambda t: t[1], reverse=True)
        truth_global_i2local_i = {
            t[0]: li for li, t in enumerate(ranked_truth_vals)
        }
        pred_order_list = [truth_global_i2local_i[t[0]] for t in ranked_pred_vals]
        max_rel_score = max(pred_order_list)
        pred_rel_list = [max_rel_score + -1 * v for v in pred_order_list]
        inst_ndcg = ndcg_at_k(pred_rel_list, k_cands)
        metrics["avg {} ndcg-{}".format(key_prefix, k_cands)] += inst_ndcg
        n_inst_dict["avg {} ndcg-{}".format(key_prefix, k_cands)] += 1

    def _add_knn_candidates(curr_i, k_cands):
        _knn_cand_indices = []
        front_idx, back_idx = curr_i - 1, curr_i + 1
        take_front = True if curr_i == len(pred_target_list)-1 else False
        while len(_knn_cand_indices) < k_cands:
            if take_front and 0 <= front_idx < len(pred_target_list):
                _knn_cand_indices.append(front_idx)
                front_idx -= 1
            elif 0 <= back_idx < len(pred_target_list):
                _knn_cand_indices.append(back_idx)
                back_idx += 1
            else:
                break
            take_front = not take_front
        assert len(_knn_cand_indices) > 0
        return _knn_cand_indices

    assert len(pred_list) == len(truth_list) > 0
    joint_list = list(zip(pred_list, truth_list))
    joint_list.sort(key=lambda t: t[1], reverse=True) # sort to make finding knn easier
    pred_target_list, truth_target_list = [t[0] for t in joint_list], [t[1] for t in joint_list]
    metrics = collections.defaultdict(float)
    n_inst_dict = collections.defaultdict(int)
    for k in ndcg_p_list:
        bar = None
        if verbose:
            bar = tqdm(total=len(pred_target_list), desc="Computing top-{} ranking results".format(k),
                       ascii=True, leave=False)
        for i, pred_val in enumerate(pred_target_list):
            random_cand_indices = set(random.sample(range(0, len(pred_target_list)), k))
            if i in random_cand_indices:
                random_cand_indices.remove(i)
            random_cand_indices = list(random_cand_indices)[:k-1]
            _compute_multi_cand_ranking_metrics(i, "random", random_cand_indices, k)
            knn_cand_indices = _add_knn_candidates(i, k-1)
            _compute_multi_cand_ranking_metrics(i, "knn", knn_cand_indices, k)
            if bar is not None: bar.update(1)
        if bar is not None: bar.close()
    for key in metrics:
        metrics[key] /= n_inst_dict[key]
    _compute_global_ranking_metrics()
    # print("Ranking results: {}".format(metrics))
    metrics["input size"] = len(pred_target_list)
    for k in top_overlap_k_list:
        metrics["top-{} overlaps".format(k)] = top_k_regression_overlap_score(pred_target_list, truth_target_list, k)
    return metrics


def top_k_regression_overlap_score(pred_list, truth_list, k):
    """
    Computes the percentage of the predicted top-k values that overlaps the truth top-k values
    """
    ranked_truth_list = [(i, val) for i, val in enumerate(truth_list)]
    ranked_truth_list.sort(key=lambda t: t[1], reverse=True)
    top_k_truth_ids = set([i for i, v in ranked_truth_list][:k])
    ranked_pred_list = [(i, val) for i, val in enumerate(pred_list)]
    ranked_pred_list.sort(key=lambda t: t[1], reverse=True)
    top_k_pred_ids = set([i for i, v in ranked_pred_list][:k])
    return len(top_k_truth_ids.intersection(top_k_pred_ids)) * 1. / k
