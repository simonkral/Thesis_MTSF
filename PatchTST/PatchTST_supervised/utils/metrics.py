import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


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


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr

def metric_of_channels(preds, trues):
    """
    Computes MAE, MSE, RSE for each channel.
    
    Returns:
        List of dicts, one per channel.
    """
    num_channels = preds.shape[-1]
    channel_metrics = []

    for c in range(num_channels):
        pred_c = preds[:, :, c]
        true_c = trues[:, :, c]

        mae = MAE(pred_c, true_c)
        mse = MSE(pred_c, true_c)
        rse = RSE(pred_c, true_c)

        channel_metrics.append({
            'channel': c,
            'mae': mae,
            'mse': mse,
            'rse': rse,
        })

    return channel_metrics
