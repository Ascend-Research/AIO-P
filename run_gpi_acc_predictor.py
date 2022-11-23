import torch
import random
import numpy as np
from params import *
import torch_geometric
from constants import *
import utils.model_utils as m_util
from utils.misc_utils import RunningStatMeter
from model_src.model_helpers import BookKeeper
from model_src.comp_graph.tf_comp_graph import OP2I
from model_src.comp_graph.tf_comp_graph_models import make_cg_regressor, make_fuzzy_cg_encoder
from model_src.predictor.gpi_family_data_manager import FamilyDataManager
from model_src.comp_graph.tf_comp_graph_dataloaders import CGRegressDataLoader
from utils.model_utils import set_random_seed, device, add_weight_decay, get_activ_by_name
from model_src.predictor.model_perf_predictor import train_predictor, run_predictor_demo
import time
import copy
from model_src.get_truth_and_preds import get_reg_truth_and_preds
from model_src.demo_functions import pure_regressor_metrics, correlation_metrics
from model_src.multitask.normalizers import MultiTaskNormalizer


"""
Naive accuracy predictor training routine
For building a generalizable predictor interface
"""



def prepare_local_params(parser, ext_args=None):
    parser.add_argument("-model_name", required=False, type=str,
                        default="GraphConv")
    parser.add_argument("-family_train", required=False, type=str,
                        default="nb101"
                        )
    parser.add_argument('-family_test', required=False, type=str,
                        default="ofa_pn"
                                "+ofa_mbv3")
    parser.add_argument("-dev_ratio", required=False, type=float,
                        default=0.1)
    parser.add_argument("-test_ratio", required=False, type=float,
                        default=0.1)
    parser.add_argument("-epochs", required=False, type=int,
                        default=40)
    parser.add_argument("-fine_tune_epochs", required=False, type=int,
                        default=100)
    parser.add_argument("-batch_size", required=False, type=int,
                        default=32)
    parser.add_argument("-initial_lr", required=False, type=float,
                        default=0.0001)
    parser.add_argument("-in_channels", help="", type=int,
                        default=32, required=False)
    parser.add_argument("-hidden_size", help="", type=int,
                        default=32, required=False)
    parser.add_argument("-out_channels", help="", type=int,
                        default=32, required=False)
    parser.add_argument("-num_layers", help="", type=int,
                        default=6, required=False)
    parser.add_argument("-dropout_prob", help="", type=float,
                        default=0.0, required=False)
    parser.add_argument("-aggr_method", required=False, type=str,
                        default="mean")
    parser.add_argument("-gnn_activ", required=False, type=str,
                        default="tanh")
    parser.add_argument("-reg_activ", required=False, type=str,
                        default=None)
    parser.add_argument("-normalize_HW_per_family", required=False, action="store_true",
                        default=False)
    parser.add_argument('-gnn_type', required=False, default="GraphConv")
    parser.add_argument('-gnn_args', required=False, default="")
    parser.add_argument('-e_chk', type=str, default=None, required=False)
    parser.add_argument('-num_seeds', type=int, default=5, required=False)
    parser.add_argument('-rescale', required=False, type=float, default=0.)
    parser.add_argument('-rs_l1', action="store_true", default=False,
                        help="Use L1 loss for re-scale learning")
    parser.add_argument('-k_adapt', required=False, type=int, default=-1)
    parser.add_argument('-family_k', required=False, type=str,
                        default="hiaml")
    parser.add_argument('-k_epochs', type=int, default=40)
    parser.add_argument('-tar_norm', type=str, default="none")

    return parser.parse_args(ext_args)


def get_family_train_size_dict(args):
    if args is None:
        return {}
    rv = {}
    for arg in args:
        if "#" in arg:
            fam, size = arg.split("#")
        else:
            fam = arg
            size = 0
        rv[fam] = int(float(size))
    return rv


def main(params):
    params.model_name = "gpi_acc_predictor_{}_seed{}".format(params.model_name, params.seed)
    book_keeper = BookKeeper(log_file_name=params.model_name + ".txt",
                             model_name=params.model_name,
                             saved_models_dir=params.saved_models_dir,
                             init_eval_perf=float("inf"), eval_perf_comp_func=lambda old, new: new < old,
                             saved_model_file=params.saved_model_file,
                             logs_dir=params.logs_dir)

    if type(params.family_test) is str:
        families_train = list(v for v in set(params.family_train.split("+")) if len(v) > 0)
        families_train.sort()
        families_test = params.family_test.split("+")
    else:
        families_train = params.family_train
        families_test = params.family_test

    book_keeper.log("Params: {}".format(params), verbose=False)
    set_random_seed(params.seed, log_f=book_keeper.log)
    book_keeper.log("Train Families: {}".format(families_train))
    book_keeper.log("Test Families: {}".format(families_test))

    families_test = get_family_train_size_dict(families_test)

    data_manager = FamilyDataManager(families_train, log_f=book_keeper.log)
    family2sets = \
        data_manager.get_regress_train_dev_test_sets(params.dev_ratio, params.test_ratio,
                                                     normalize_HW_per_family=params.normalize_HW_per_family,
                                                     normalize_target=False, group_by_family=True)

    train_data, dev_data, test_data = [], [], []
    for f, (fam_train, fam_dev, fam_test) in family2sets.items():
        train_data.extend(fam_train)
        dev_data.extend(fam_dev)
        test_data.extend(fam_test)

    train_norm = MultiTaskNormalizer(data=train_data, type=params.tar_norm)
    train_data = train_norm.transform(train_data)

    random.shuffle(train_data)
    random.shuffle(dev_data)
    random.shuffle(test_data)
    book_keeper.log("Train size: {}".format(len(train_data)))
    book_keeper.log("Dev size: {}".format(len(dev_data)))
    book_keeper.log("Test size: {}".format(len(test_data)))

    train_loader = CGRegressDataLoader(params.batch_size, train_data,)
    dev_loader = CGRegressDataLoader(params.batch_size, dev_data,)
    test_loader = CGRegressDataLoader(params.batch_size, test_data,)

    book_keeper.log(
        "{} overlap(s) between train/dev loaders".format(train_loader.get_overlapping_data_count(dev_loader)))
    book_keeper.log(
        "{} overlap(s) between train/test loaders".format(train_loader.get_overlapping_data_count(test_loader)))
    book_keeper.log(
        "{} overlap(s) between dev/test loaders".format(dev_loader.get_overlapping_data_count(test_loader)))

    book_keeper.log("Initializing {}".format(params.model_name))

    def gnn_constructor(in_channels, out_channels):
        return eval("torch_geometric.nn.%s(%d, %d, %s)"
                    % (params.gnn_type, in_channels, out_channels, params.gnn_args))

    model = make_cg_regressor(n_unique_labels=len(OP2I().build_from_file()), out_embed_size=params.in_channels,
                              shape_embed_size=8, kernel_embed_size=8, n_unique_kernels=8, n_shape_vals=6,
                              hidden_size=params.hidden_size, out_channels=params.out_channels,
                              gnn_constructor=gnn_constructor,
                              gnn_activ=get_activ_by_name(params.gnn_activ), n_gnn_layers=params.num_layers,
                              dropout_prob=params.dropout_prob, aggr_method=params.aggr_method,
                              regressor_activ=get_activ_by_name(params.reg_activ)).to(device())

    if params.e_chk is not None:
        if "encoder" in params.e_chk:
            encoder = make_fuzzy_cg_encoder(n_unique_labels=len(OP2I().build_from_file()), out_embed_size=params.in_channels,
                                            shape_embed_size=8, kernel_embed_size=8, n_unique_kernels=8, n_shape_vals=6,
                                            hidden_size=params.hidden_size, out_channels=params.out_channels,
                                            gnn_constructor=gnn_constructor,
                                            gnn_activ=get_activ_by_name(params.gnn_activ), n_gnn_layers=params.num_layers,
                                            dropout_prob=0, aggr_method=params.aggr_method).to(device())

            book_keeper.load_model_checkpoint(encoder, checkpoint_file=params.e_chk, skip_eval_perfs=True,
                                              allow_silent_fail=False)

            model.embed_layer = encoder.embed_layer
            model.encoder.gnn_layers = encoder.encoder.gnn_layers

        else:
            book_keeper.load_model_checkpoint(model, allow_silent_fail=False, skip_eval_perfs=True,
                                              checkpoint_file=params.e_chk)

    perf_criterion = torch.nn.MSELoss()
    model_params = add_weight_decay(model, weight_decay=0.)
    optimizer = torch.optim.Adam(model_params, lr=params.initial_lr)

    book_keeper.log(model)
    book_keeper.log("Model name: {}".format(params.model_name))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    book_keeper.log("Number of trainable parameters: {}".format(n_params))

    reg_metrics = ["MSE", "MAE", "MAPE"]

    def _batch_fwd_func(_model, _batch):
        # Define how a batch is handled by the model
        regular_node_inds = _batch[DK_BATCH_CG_REGULAR_IDX]
        regular_node_shapes = _batch[DK_BATCH_CG_REGULAR_SHAPES]
        weighted_node_inds = _batch[DK_BATCH_CG_WEIGHTED_IDX]
        weighted_node_shapes = _batch[DK_BATCH_CG_WEIGHTED_SHAPES]
        weighted_node_kernels = _batch[DK_BATCH_CG_WEIGHTED_KERNELS]
        weighted_node_bias = _batch[DK_BATCH_CG_WEIGHTED_BIAS]
        edge_tsr_list = _batch[DK_BATCH_EDGE_TSR_LIST]
        batch_last_node_idx_list = _batch[DK_BATCH_LAST_NODE_IDX_LIST]

        return _model(regular_node_inds, regular_node_shapes,
                      weighted_node_inds, weighted_node_shapes, weighted_node_kernels, weighted_node_bias,
                      edge_tsr_list, batch_last_node_idx_list, ext_feat=[0, 0])

    book_keeper.log("Training for {} epochs".format(params.epochs))
    start = time.time()
    try:
        train_predictor(_batch_fwd_func, model, train_loader, perf_criterion, optimizer, book_keeper,
                        num_epochs=params.epochs, max_gradient_norm=params.max_gradient_norm,
                        dev_loader=dev_loader, model_name=params.model_name)
    except KeyboardInterrupt:
        book_keeper.log("Training interrupted")

    book_keeper.report_curr_best()
    book_keeper.load_model_checkpoint(model, allow_silent_fail=True, skip_eval_perfs=True,
                                      checkpoint_file=P_SEP.join([book_keeper.saved_models_dir,
                                                                  params.model_name + "_best.pt"]))
    end = time.time()
    with torch.no_grad():
        model.eval()
        book_keeper.log("===============Predictions===============")
        run_predictor_demo(_batch_fwd_func, model, test_loader,
                           n_batches=10, log_f=book_keeper.log)
        book_keeper.log("===============Overall Test===============")
        test_labels, test_preds, test_flops = get_reg_truth_and_preds(model, test_loader, _batch_fwd_func, printfunc=book_keeper.log)
        test_preds = train_norm.inverse(test_preds, test_flops)
        test_reg_metrics = pure_regressor_metrics(test_labels, test_preds, printfunc=book_keeper.log)
        for i, metric in enumerate(reg_metrics):
            book_keeper.log("Test {}: {}".format(metric, test_reg_metrics[i]))
            if metric is "MAE":
                metric_list = ["Train MAE"]
                results_list = [test_reg_metrics[i]]

        [overall_sp_rho] = correlation_metrics(test_labels, test_preds, printfunc=book_keeper.log)

        book_keeper.log("Total time: %s" % (end - start))
        metric_list.append("Train SRCC")
        results_list.append(overall_sp_rho)
    metric_list = []
    results_list = []

    ft_k = 0
    if params.k_adapt >= 0:
        from model_src.multitask.k_adapters import CGRegressorAdapter
        book_keeper.log("Train k-Adapter on family {}".format(params.family_k))
        for param in model.parameters():
            param.requires_grad = False

        families_k = list(v for v in params.family_k.split("+") if len(v) > 0)
        print("Families_k")
        print(families_k)
        book_keeper.log("K-Adapter families: {}".format(families_k))
        k_dm = FamilyDataManager(families_k, log_f=book_keeper.log)

        # First 2 args are ratios for the code demo with limited data; we used 0.05 for both in our experiments
        k_fam2sets = k_dm.get_regress_train_dev_test_sets(0.25, 0.25,
        normalize_HW_per_family=params.normalize_HW_per_family, normalize_target=False, group_by_family=True)

        book_keeper.log("Creating k-Adapter with existing model")
        model = CGRegressorAdapter(model, K=len(families_k), ft_adapter=params.k_adapt)

        k = 0
        for f, (fam_train, fam_dev, fam_test) in k_fam2sets.items():
            model.set_k(k)
            book_keeper.log("Train K-Adapter on {}".format(f))
            k_norm = MultiTaskNormalizer(data=fam_train, type=params.tar_norm)
            fam_train = k_norm.transform(fam_train)

            random.shuffle(fam_train)
            k_train_loader = CGRegressDataLoader(params.batch_size, fam_train)
            k_dev_loader = CGRegressDataLoader(params.batch_size, fam_dev)
            k_test_loader = CGRegressDataLoader(params.batch_size, fam_test)

            k_opt = torch.optim.Adam(model.parameters(), lr=params.initial_lr)
            k_params_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
            book_keeper.log("Trainable K-Adapter parameters: {}".format(k_params_grad))
            book_keeper.log("Training for {} epochs:".format(params.k_epochs))

            train_predictor(_batch_fwd_func, model, k_train_loader, perf_criterion, k_opt, book_keeper,
                            num_epochs=params.k_epochs, max_gradient_norm=params.max_gradient_norm,
                            dev_loader=k_dev_loader, model_name=params.model_name, checkpoint=False)

            book_keeper.log("k-Adapter Distribution for family {}".format(f))
            k_labels, k_preds, k_flops = get_reg_truth_and_preds(model, k_test_loader, _batch_fwd_func, printfunc=book_keeper.log)

            k_preds = k_norm.inverse(k_preds, k_flops)

            k_adapt_mae = pure_regressor_metrics(k_labels, k_preds, printfunc=book_keeper.log)[1]
            metric_list.append("K-Adapt {} MAE".format(f))
            results_list.append(k_adapt_mae)
            book_keeper.log("K-Adapter {} SRCC".format(f))
            [k_srcc] = correlation_metrics(k_labels, k_preds, printfunc=book_keeper.log)
            metric_list.append("K-Adapt {} SRCC".format(f))
            results_list.append(k_srcc)
        
            k += 1
        model.set_k(-1)
        book_keeper.checkpoint_model("_k{}.pt".format(k), params.k_epochs, model, k_opt)
        ft_k = k

    foreign_families = tuple(families_test.keys())
    book_keeper.log("Starting fine-tune on foreign families: {}".format(foreign_families))

    ff_manager = FamilyDataManager(families=foreign_families, log_f=book_keeper.log)
    ff_data = ff_manager.get_regress_train_dev_test_sets(0, 1.0,
                                                         group_by_family=True,
                                                         normalize_HW_per_family=params.normalize_HW_per_family,
                                                         normalize_target=False) #, all_test=True)

    for family, size in families_test.items():
        foreign_data = ff_data[family][-1] + ff_data[family][-2]

        foreign_data.sort(key=lambda x: x[0].name, reverse=False)


        np.random.seed(params.seed)
        np.random.shuffle(foreign_data)

        test_size = len(foreign_data) - size
        fine_tune_data = foreign_data[test_size:]
        foreign_test_data = foreign_data[:test_size]

        norm_data = fine_tune_data if size > 0 else foreign_test_data
        foreign_norm = MultiTaskNormalizer(norm_data, type=params.tar_norm)

        book_keeper.log("Foreign family {} fine-tune size: {}".format(family, len(fine_tune_data)))
        book_keeper.log("Foreign family {} test size: {}".format(family, len(foreign_test_data)))
        foreign_test_loader = CGRegressDataLoader(1, foreign_test_data)

        if len(fine_tune_data) > 0:
            fine_tune_data = foreign_norm.transform(fine_tune_data)

            ft_model = copy.deepcopy(model)
            perf_criterion_ft = perf_criterion
            if params.rescale > 0.:
                from model_src.get_truth_and_preds import RescaleLoss
                book_keeper.log("Fine-tune using rescaling factors. Disabling grad for existing params")
                for param in ft_model.parameters():
                    param.requires_grad = False
                book_keeper.log("Instantiating alpha and b")
                ft_model.init_alpha()
                rescale_criterion = torch.nn.L1Loss() if params.rs_l1 else torch.nn.MSELoss()
                perf_criterion_ft = RescaleLoss(rescale_criterion, ft_model, params.rescale)
                rescale_params = [ft_model.alpha, ft_model.b]
                ft_opt = torch.optim.Adam(rescale_params, lr=params.initial_lr)  # / 10)
            elif params.k_adapt >= 0:
                book_keeper.log("k-Adapter ready for fine-tuning!")
                ft_model.set_k(-1)
                ft_params = ft_model.ft_parameters()
                ft_opt = torch.optim.Adam(ft_params, lr=params.initial_lr)
                ft_params_grad = sum(p.numel() for p in ft_params if p.requires_grad)
                book_keeper.log("FT trainable parameters: {}".format(ft_params_grad))
            else:
                ft_opt = torch.optim.Adam(ft_model.parameters(), lr=params.initial_lr)

            ft_loader = CGRegressDataLoader(1, fine_tune_data)

            book_keeper.log("Fine-tuning for {} epochs".format(params.fine_tune_epochs))
            train_predictor(_batch_fwd_func, ft_model, ft_loader, perf_criterion_ft, ft_opt, book_keeper,
                            num_epochs=params.fine_tune_epochs, max_gradient_norm=params.max_gradient_norm,
                            dev_loader=None, checkpoint=False)

        with torch.no_grad():
            model.eval()
            foreign_labels, foreign_preds, foreign_flops = get_reg_truth_and_preds(model, foreign_test_loader, _batch_fwd_func, printfunc=book_keeper.log)

            foreign_preds = foreign_norm.inverse(foreign_preds, foreign_flops)
            test_reg_metrics = pure_regressor_metrics(foreign_labels, foreign_preds, printfunc=book_keeper.log)
            for i, metric in enumerate(reg_metrics):
                book_keeper.log("{}-NoFT {}: {}".format(family, metric, test_reg_metrics[i]))
                if metric is "MAE":
                    metric_list.append("{}-NoFT MAE".format(family))
                    results_list.append(test_reg_metrics[i])

            [no_ft_sp] = correlation_metrics(foreign_labels, foreign_preds, printfunc=book_keeper.log)
            book_keeper.log("{}-NoFT Spearman Rho: {}".format(family, no_ft_sp))
            metric_list.append("{}-NoFT SRCC".format(family))
            results_list.append(no_ft_sp)

            book_keeper.log("Total time: %s" % (end - start))

            if len(fine_tune_data) > 0 and params.fine_tune_epochs > 0:
                chkpt_str = "_{}_ft.pt".format(family)
                if ft_k > 0:
                    chkpt_str = chkpt_str.replace("ft.pt", "ft_k{}.pt".format(ft_k))
                if params.rescale > 0.:
                    chkpt_str = chkpt_str.replace("_ft_", "_scale_ft_")
                book_keeper.checkpoint_model(chkpt_str, params.fine_tune_epochs, ft_model, ft_opt)

                foreign_labels, foreign_preds, foreign_flops = get_reg_truth_and_preds(ft_model, foreign_test_loader, _batch_fwd_func,
                                                                                       printfunc=book_keeper.log)
                foreign_preds = foreign_norm.inverse(foreign_preds, foreign_flops)
                test_reg_metrics = pure_regressor_metrics(foreign_labels, foreign_preds, printfunc=book_keeper.log)
                for i, metric in enumerate(reg_metrics):
                    book_keeper.log("{}-FT {}: {}".format(family, metric, test_reg_metrics[i]))
                    if metric is "MAE":
                        metric_list.append("{}-FT MAE".format(family))
                        results_list.append(test_reg_metrics[i])

                [ft_sp] = correlation_metrics(foreign_labels, foreign_preds, printfunc=book_keeper.log)
                book_keeper.log("{}-FT Spearman Rho: {}".format(family, ft_sp))
                metric_list.append("{}-FT SRCC".format(family))
                results_list.append(ft_sp)

    return metric_list, results_list


if __name__ == "__main__":
    _parser = prepare_global_params()
    params = prepare_local_params(_parser)
    m_util.DEVICE_STR_OVERRIDE = params.device_str

    if params.num_seeds == 1:
        main(params)
    else:
        original_model_name = params.model_name
        book_keeper = BookKeeper(log_file_name=original_model_name + "_acc_allseeds.txt",
                                 model_name=params.model_name,
                                 logs_dir=params.logs_dir)

        book_keeper.log("Params: {}".format(params), verbose=False)
        metrics_dict = {'Mean': np.mean,
                        'S.Dev': np.std,
                        'Max': np.max,
                        'Min': np.min}

        all_results = []
        for i in range(params.num_seeds):
            params.seed = SEEDS_RAW[i % len(SEEDS_RAW)]
            if params.num_seeds > len(SEEDS_RAW):
                params.seed += i
            params.model_name = original_model_name
            metric_list, result_list = main(params)
            all_results.append(result_list)

        result_mat = np.matrix(all_results)
        banner_msg = ", ".join(metric_list)

        for i, metric in enumerate(metric_list):
            book_keeper.log(metric)
            for measure in metrics_dict.keys():
                computed_metric = metrics_dict[measure](result_mat[:, i]).squeeze()
                book_keeper.log("%s: %.6f" % (measure, computed_metric))
