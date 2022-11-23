import time
import torch
import collections
from tqdm import tqdm
from constants import *
from utils.model_utils import device
from utils.eval_utils import get_regression_metrics, get_regression_rank_metrics
from model_src.strategies.util import compute_accuracy


def train_predictor(batch_fwd_func, model, train_loader, criterion, optimizer, book_keeper, num_epochs,
                    max_gradient_norm=5.0, eval_start_epoch=1, eval_every_epoch=1,
                    rv_metric_name="mean_absolute_percent_error", completed_epochs=0,
                    dev_loader=None, checkpoint=True, model_name='GraphConv', encoder=False, context=False,
                    masking=False):
    model = model.to(device())
    if masking:
        for crit in criterion:
            crit = crit.to(device())
    elif criterion is not None:
        criterion = criterion.to(device())
    for epoch in range(num_epochs):
        report_epoch = epoch + completed_epochs + 1
        model.train()

        if encoder:
            train_score = run_encoder_epoch(batch_fwd_func, model, train_loader, optimizer, book_keeper,
                                            max_grad_norm=max_gradient_norm, curr_epoch=report_epoch)
        elif context:
            train_score = run_context_epoch(batch_fwd_func, model, train_loader, criterion, optimizer, book_keeper,
                                            max_grad_norm=max_gradient_norm, curr_epoch=report_epoch)
        elif masking:
            train_score = run_masking_epoch(batch_fwd_func, model, train_loader, criterion, optimizer, book_keeper,
                                            max_grad_norm=max_gradient_norm, curr_epoch=report_epoch)
        else:
            train_score = run_predictor_epoch(batch_fwd_func, model, train_loader, criterion, optimizer, book_keeper,
                                              rv_metric_name=rv_metric_name, max_grad_norm=max_gradient_norm,
                                              curr_epoch=report_epoch, model_name=model_name)
        book_keeper.log("Train score at epoch {}: {}".format(report_epoch, train_score))
        if checkpoint:
            if encoder:
                book_keeper.checkpoint_model("_encoder_latest.pt", report_epoch, model, optimizer)
            elif context:
                book_keeper.checkpoint_model("_context_latest.pt", report_epoch, model, optimizer)
            elif masking:
                book_keeper.checkpoint_model("_masking_latest.pt", report_epoch, model, optimizer)
            else:
                book_keeper.checkpoint_model("_latest.pt", report_epoch, model, optimizer)

        if dev_loader is not None:
            with torch.no_grad():
                model.eval()
                if report_epoch >= eval_start_epoch and report_epoch % eval_every_epoch == 0:
                    if encoder:
                        model.train()
                        dev_score = run_encoder_epoch(batch_fwd_func, model, dev_loader, None, book_keeper,
                                                      curr_epoch=report_epoch, desc="Dev")
                    elif context:
                        dev_score = run_context_epoch(batch_fwd_func, model, dev_loader, criterion, None, book_keeper,
                                                      desc="Dev", max_grad_norm=max_gradient_norm,
                                                      curr_epoch=report_epoch)
                    elif masking:
                        dev_score = run_masking_epoch(batch_fwd_func, model, dev_loader, criterion, None, book_keeper,
                                                      desc="Dev", max_grad_norm=max_gradient_norm,
                                                      curr_epoch=report_epoch)
                    else:
                        dev_score = run_predictor_epoch(batch_fwd_func, model, dev_loader, criterion, None, book_keeper,
                                                        rv_metric_name=rv_metric_name, desc="Dev",
                                                        max_grad_norm=max_gradient_norm,
                                                        curr_epoch=report_epoch)
                    book_keeper.log("Dev score at epoch {}: {}".format(report_epoch, dev_score))
                    if not encoder:
                        if checkpoint:
                            book_keeper.checkpoint_model("_best.pt", report_epoch, model, optimizer,
                                                         eval_perf=dev_score)
                            book_keeper.report_curr_best()
        book_keeper.log("")


def run_predictor_epoch(batch_fwd_func, model, loader, criterion, optimizer, book_keeper,
                        desc="Train", curr_epoch=0, max_grad_norm=5.0, report_metrics=True,
                        rv_metric_name="mean_absolute_percent_error", model_name='GraphConv'):
    """
    Compatible with a predictor/loader that batches same-sized graphs
    """
    start = time.time()
    total_loss, n_instances = 0., 0
    metrics_dict = collections.defaultdict(float)
    preds, targets = [], []
    for batch in tqdm(loader, desc=desc, ascii=True):
        batch_vals = batch_fwd_func(model, batch)
        truth = batch[DK_BATCH_TARGET_TSR].to(device())
        pred = batch_vals.squeeze(1)
        loss = criterion(pred, truth)

        if model.training and check_supergat_conditions(model, model_name):
            if hasattr(model, 'supergat_encoder'):
                sg_module = model.supergat_encoder.encoder
            else:
                sg_module = model.encoder
            att_loss = 0.0

            for supergat in sg_module.gnn_layers:
                att_loss += supergat.get_attention_loss()

            # Add attention edge loss to formal loss
            loss += 0.001 * att_loss
        total_loss += loss.item() * batch[DK_BATCH_SIZE]
        preds.extend(pred.detach().view(-1).tolist())
        targets.extend(truth.detach().view(-1).tolist())
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        n_instances += batch[DK_BATCH_SIZE]
    elapsed = time.time() - start
    rv_loss = total_loss / n_instances
    msg = desc + " epoch: {}, loss: {}, elapsed time: {}".format(curr_epoch, rv_loss, elapsed)
    book_keeper.log(msg)
    if report_metrics:
        metrics_dict = get_regression_metrics(preds, targets)
        rank_metrics = get_regression_rank_metrics(preds, targets,
                                                   top_overlap_k_list=(5, 10, 25, 50),
                                                   verbose=True)
        metrics_dict["spearman_rho"] = rank_metrics["spearman_rho"]
        metrics_dict["spearman_p"] = rank_metrics["spearman_p"]
        metrics_dict["top-5 overlaps"] = rank_metrics["top-5 overlaps"]
        metrics_dict["top-10 overlaps"] = rank_metrics["top-10 overlaps"]
        metrics_dict["top-25 overlaps"] = rank_metrics["top-25 overlaps"]
        metrics_dict["top-50 overlaps"] = rank_metrics["top-50 overlaps"]
        book_keeper.log("{} performance: {}".format(desc, str(metrics_dict)))
    return rv_loss if not report_metrics else metrics_dict[rv_metric_name]


def check_supergat_conditions(model, model_name):
    if hasattr(model, 'supergat_encoder'):
        return False # Changed this as self-supervising the supergat doesn't seem to be good
    elif 'supergat' in model_name.lower():
        return True
    else:
        return False


def run_encoder_epoch(batch_fwd_func, encoder, loader, optimizer, book_keeper, desc="Train", curr_epoch=0,
                      max_grad_norm=5.0):
    """
    Compatible with a predictor/loader that batches same-sized graphs
    """
    start = time.time()
    total_loss, n_instances = 0., 0
    for batch in tqdm(loader, desc=desc, ascii=True):
        _ = batch_fwd_func(encoder, batch)
        # We just compute the SuperGAT loss
        att_loss = 0.0
        # Loop over GNN layers in encoder
        for supergat in encoder.encoder.gnn_layers:
            att_loss += supergat.get_attention_loss()
        total_loss += att_loss.item() * batch[DK_BATCH_SIZE]
        if optimizer is not None:
            optimizer.zero_grad()
            att_loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_grad_norm)
            optimizer.step()
        n_instances += batch[DK_BATCH_SIZE]
    elapsed = time.time() - start
    rv_loss = total_loss / n_instances
    msg = desc + " epoch: {}, att_loss: {}, elapsed time: {}".format(curr_epoch, rv_loss, elapsed)
    book_keeper.log(msg)
    return rv_loss


def run_context_epoch(batch_fwd_func, encoder, loader, criterion, optimizer, book_keeper, desc="Train", curr_epoch=0,
                      max_grad_norm=5.0):
    """
    Compatible with a predictor/loader that batches same-sized graphs
    """
    start = time.time()
    balanced_loss_accum, acc_accum = 0, 0
    for step, batch in enumerate(tqdm(loader, desc=desc, ascii=True)):
        pred_pos, pred_neg = batch_fwd_func(encoder, batch)
        loss_pos = criterion(pred_pos.double(), torch.ones(len(pred_pos)).to(pred_pos.device).double())
        loss_neg = criterion(pred_neg.double(), torch.zeros(len(pred_neg)).to(pred_neg.device).double())
        loss = loss_pos + encoder.ns * loss_neg
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_grad_norm)
            optimizer.step()

        balanced_loss_accum += float(loss_pos.detach().cpu().item() + loss_neg.detach().cpu().item())
        acc_accum += 0.5 * (float(torch.sum(pred_pos > 0).detach().cpu().item())/len(pred_pos) +
                            float(torch.sum(pred_neg < 0).detach().cpu().item())/len(pred_neg))
    elapsed = time.time() - start
    balanced_loss_accum /= step
    acc_accum /= step
    msg = desc + " epoch: {}, balanced_loss: {}, acc: {}, elapsed time: {}".format(curr_epoch, balanced_loss_accum,
                                                                                   acc_accum, elapsed)
    book_keeper.log(msg)
    return acc_accum


def run_masking_epoch(batch_fwd_func, encoder, loader, criterions, optimizer, book_keeper, desc="Train", curr_epoch=0,
                      max_grad_norm=5.0):

    start = time.time()
    loss_accum, acc_op_accum, shape_mse = 0, 0, 0
    num_samples = 0
    for step, batch in enumerate(tqdm(loader, desc=desc, ascii=True)):
        batch, pred_nodes, pred_shapes = batch_fwd_func(encoder, batch)

        # Compute loss on node operation type
        num_samples += len(pred_nodes)
        loss = criterions[0](pred_nodes, batch.mask_node_ops)
        acc_op = compute_accuracy(pred_nodes, batch.mask_node_ops)
        acc_op_accum += acc_op

        # Compute loss on node shapes
        shape_error = criterions[1](pred_shapes, batch.mask_node_shapes)
        loss += shape_error
        shape_mse += shape_error

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_grad_norm)
            optimizer.step()

        loss_accum += float(loss.cpu().item())

    elapsed = time.time() - start
    loss_accum /= num_samples
    acc_op_accum /= num_samples
    shape_mse /= num_samples
    msg = desc + " epoch: {}, loss: {}, node op acc: {}, shape mse: {}, elapsed time: {}".format(curr_epoch,
                                                                                                 loss_accum,
                                                                                                 acc_op_accum,
                                                                                                 shape_mse,
                                                                                                 elapsed)
    book_keeper.log(msg)
    return acc_op_accum


def run_predictor_demo(batch_fwd_func, model, loader, log_f=print,
                       n_batches=1, normalize_constant=None,
                       input_str_key=None):
    n_visited = 0
    input_str_list = []
    preds, targets = [], []
    for batch in loader:
        if n_visited == n_batches:
            break
        batch_vals = batch_fwd_func(model, batch)
        truth = batch[DK_BATCH_TARGET_TSR].to(device())
        pred = batch_vals.squeeze(1)
        pred_list = pred.detach().tolist()
        target_list = truth.detach().tolist()
        preds.extend(pred_list)
        targets.extend(target_list)
        n_visited += 1
        if input_str_key is not None:
            batch_input_str = batch[input_str_key]
            for bi in range(len(pred_list)):
                input_str_list.append(batch_input_str[bi])
    for i, pred in enumerate(preds):
        input_str = input_str_list[i] if len(input_str_list) == len(preds) else None
        if input_str is not None:
            log_f("Input: {}".format(input_str))
        log_f("Pred raw: {}".format(pred))
        log_f("Truth raw: {}".format(targets[i]))
        if normalize_constant is not None:
            log_f("Normalize constant: {}".format(normalize_constant))
            log_f("Pred un-normalized: {:.3f}".format(pred * normalize_constant))
            log_f("Truth un-normalized: {:.3f}".format(targets[i] * normalize_constant))
        log_f("")

