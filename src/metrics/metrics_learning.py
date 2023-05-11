import pandas as pd
import torch

from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

import src.constants as cst
import numpy as np


def compute_sk_cm(truths, predictions):
    y_actu = pd.Series(truths, name='actual')
    y_pred = pd.Series(predictions, name='predicted')
    mat_confusion = confusion_matrix(y_actu, y_pred)
    return mat_confusion


def compute_metrics(ys, predictions, model_step, loss_vals, si):
    ys = torch.Tensor(ys)
    predictions = torch.Tensor(predictions)

    cr = classification_report(ys, predictions, output_dict=True, zero_division=0)
    accuracy = cr['accuracy']  # MICRO-F1

    f1score = cr['macro avg']['f1-score']  # MACRO-F1
    precision = cr['macro avg']['precision']  # MACRO-PRECISION
    recall = cr['macro avg']['recall']  # MACRO-RECALL

    f1score_w = cr['weighted avg']['f1-score']  # WEIGHTED-F1
    precision_w = cr['weighted avg']['precision']  # WEIGHTED-PRECISION
    recall_w = cr['weighted avg']['recall']  # WEIGHTED-RECALL

    mcc = matthews_corrcoef(ys, predictions)
    cok = cohen_kappa_score(ys, predictions)

    val_dict = {
        model_step.value + f"_{si}_" + cst.Metrics.F1.value: float(f1score),
        model_step.value + f"_{si}_" + cst.Metrics.F1_W.value: float(f1score_w),

        model_step.value + f"_{si}_" + cst.Metrics.PRECISION.value: float(precision),
        model_step.value + f"_{si}_" + cst.Metrics.PRECISION_W.value: float(precision_w),

        model_step.value + f"_{si}_" + cst.Metrics.RECALL.value: float(recall),
        model_step.value + f"_{si}_" + cst.Metrics.RECALL_W.value: float(recall_w),

        model_step.value + f"_{si}_" + cst.Metrics.ACCURACY.value: float(accuracy),
        model_step.value + f"_{si}_" + cst.Metrics.MCC.value: float(mcc),
        model_step.value + f"_{si}_" + cst.Metrics.COK.value: float(cok),
        # single
        model_step.value + f"_" + cst.Metrics.LOSS.value: float(np.sum(loss_vals)),
    }
    return val_dict
