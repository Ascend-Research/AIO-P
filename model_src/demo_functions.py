import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import pearsonr, spearmanr


def pure_regressor_metrics(targets, preds, printfunc=print):

    target_mse = mean_squared_error(targets, preds)
    target_mae = mean_absolute_error(targets, preds)
    target_mape = mean_absolute_percentage_error(targets, preds)
    printfunc("Pure regression metrics.\nMSE = {}\nMAE = {}\nMAPE = {}".format(target_mse, target_mae, target_mape))

    return [target_mse, target_mae, target_mape]


# Spearmann Rank Correlation - SRCC
# Pearson Correlation
def correlation_metrics(targets, preds, tau=0, pearson=False, p=False, printfunc=print):

    metrics = []
    if pearson:
        # Pearson Correlation
        pcc, pp = pearsonr(targets, preds)
        printfunc("Pearson Correlation = {}; p = {}".format(pcc, pp))
        metrics.append(pcc)
        if p:
            metrics.append(pp)

    # Spearman Correlation
    srcc, sp = spearmanr(targets, preds)
    printfunc("Spearman Correlation = {}; p = {}".format(srcc, sp))
    metrics.append(srcc)
    if p:
        metrics.append(sp)

    return metrics
