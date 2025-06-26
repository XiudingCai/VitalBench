import numpy as np
from sklearn.metrics import r2_score, mean_absolute_percentage_error, explained_variance_score
from scipy.stats import spearmanr


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def R2(pred, true):
    r2 = r2_score(true, pred)
    return r2


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def metric_with_mask(pred, true):
    # mask
    mask = true != 0
    pred = pred[mask]
    true = true[mask]
    # compute
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe


def calculate_mape(pred, true):
    epsilon = 1e-8  # 添加一个小的正数，避免除以零的情况
    abs_diff = np.abs(true - pred)
    abs_true = np.abs(true)
    return np.mean(np.divide(abs_diff, abs_true + epsilon))


def AdjustedR2(pred, true, n, p):
    r2 = 1 - ((1 - r2_score(true, pred)) * (n - 1)) / (n - p - 1)
    return r2


def calculate_r2(pred, true):
    # 计算 R-squared
    numerator = np.sum((true - pred) ** 2, axis=1)
    denominator = np.sum((true - np.mean(true, axis=1, keepdims=True)) ** 2, axis=1)
    r2_per_time = 1 - numerator / denominator
    return np.mean(r2_per_time)


# def explained_variance_score(y_true, y_pred):
#     """
#     计算解释方差分数
#     :param y_true: 实际值，形状为 (N, T) 的数组
#     :param y_pred: 预测值，形状为 (N, T) 的数组
#     :return: 解释方差分数
#     """
#     # 总方差
#     total_variance = np.var(y_true, axis=0)
#
#     # 残差方差
#     residual_variance = np.var(y_true - y_pred, axis=0)
#
#     # 解释方差
#     evar = 1 - (residual_variance / total_variance)
#
#     # 返回每个输出变量的平均解释方差分数
#     return np.mean(evar)

def vm_metric_with_mask(pred, true):
    n, t, p = pred.shape
    pred = pred.transpose((2, 0, 1))
    true = true.transpose((2, 0, 1))

    pred = pred.reshape(pred.shape[0], -1)
    true = true.reshape(true.shape[0], -1)
    # B, T, C
    # print(f">>> {pred.shape} {true.shape}")
    # mask
    pred = np.where(true == 0, 0, pred)

    # r2 = R2(pred, true)
    # print(pred.shape, true.shape)
    # r2 = r2_score(pred, true)
    # r2 = calculate_r2(pred, true)

    # if r2 < -100:
    #     print(r2)
    # r2_list = []
    # for i in range(p):
    #     mask = true[i] != 0
    #     if mask.sum() != 0:
    #         r2_list.append(r2_score(pred[i][mask], true[i][mask], force_finite=True))
    # r2 = np.mean(r2_list)
    # r2 = AdjustedR2(pred, true, n, p)
    # r2 = explained_variance_score(true, pred)
    # evar = explained_variance_score(true, pred, multioutput='variance_weighted')
    r2 = r2_score(true, pred, multioutput='variance_weighted')
    # print(r2)
    # print(mae)
    # if r2 < 0:
    #     print(r2)
    #     print(pred[:10])
    #     print(true[:10])

    mask = true != 0
    pred = pred[mask]
    true = true[mask]
    # print(f"after mask{pred.shape} {true.shape}")

    # compute metrics
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    mape = np.mean(np.abs((pred - true) / true))
    cc, _ = spearmanr(true, pred)

    # mape = mean_absolute_percentage_error(true, pred)
    # print(mape1, mape)
    # r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - np.mean(true)) ** 2)
    # r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - np.mean(true, axis=-1, keepdims=True)) ** 2)
    #

    return mae, rmse, mape, r2, cc
    # return mae, rmse, r2, cc
