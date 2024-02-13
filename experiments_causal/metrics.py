"""Python script to calculate balanced accuracy."""
import torch
import pandas as pd
import numpy as np


def binary_stat_scores_format(
        target: pd.Series | np.array,
        prediction: np.array,
        threshold: float = 0.5) -> (torch.tensor, torch.tensor):
    """Transform inputs into binary torch.

    Parameters
    ----------
    target : pd.Series | np.array
        Binary target of prediction task.
    prediction : np.array
        Prediction.
    threshold : float, optional
        Threshold to convert floating predictions to binary. The default is 0.5.

    Returns
    -------
    target : torch.tensor
        Binary target of prediction task.
    prediction : torch.tensor
        Binary prediction.

    """
    if isinstance(target, pd.Series):
        target = target.to_numpy()
    target = torch.from_numpy(target)
    prediction = torch.from_numpy(prediction)
    if prediction.is_floating_point():
        if not torch.all((prediction >= 0) * (prediction <= 1)):
            # prediction is logits, convert with sigmoid
            prediction = prediction.sigmoid()
        prediction = (prediction > threshold)*1.0

    prediction = prediction.reshape(prediction.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    return target, prediction


def balanced_accuracy_score(
        target: pd.Series | np.array,
        prediction: np.array,) -> (torch.number, torch.number):
    """Compute balanced accuracy and its standard error.

    Parameters
    ----------
    target : pd.Series | np.array
        Binary target of prediction task.
    prediction : np.array
        Prediction.

    Returns
    -------
    torch.number
        Balanced accuracy.
    torch.number
        Standard error of balanced accuracy.

    """
    target, prediction = binary_stat_scores_format(target, prediction)
    n = len(prediction)
    tp = torch.count_nonzero((target == 1) & (prediction == 1))
    fn = torch.count_nonzero((target == 1) & (prediction == 0))
    tn = torch.count_nonzero((target == 0) & (prediction == 0))
    fp = torch.count_nonzero((target == 0) & (prediction == 1))
    sensitivity = tp / (tp+fn)
    specificity = tn / (tn+fp)
    balanced_acc = (sensitivity + specificity)/2
    sensitivity_se = torch.sqrt(sensitivity*(1-sensitivity)/n)
    specificity_se = torch.sqrt(specificity*(1-specificity)/n)
    balanced_acc_se = torch.sqrt(sensitivity_se**2/4 + specificity_se**2/4)
    return balanced_acc.item(), balanced_acc_se.item()
