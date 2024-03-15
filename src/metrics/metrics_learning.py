import torch

from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix

import src.constants as cst
import numpy as np


def compute_metrics(truth, prediction, loss_vals):
    truth = torch.Tensor(truth)
    prediction = torch.Tensor(prediction)

    cr = classification_report(truth, prediction, output_dict=True, zero_division=0)
    accuracy = cr['accuracy']  # MICRO-F1

    f1score = cr['macro avg']['f1-score']  # MACRO-F1
    precision = cr['macro avg']['precision']  # MACRO-PRECISION
    recall = cr['macro avg']['recall']  # MACRO-RECALL

    f1score_w = cr['weighted avg']['f1-score']  # WEIGHTED-F1
    precision_w = cr['weighted avg']['precision']  # WEIGHTED-PRECISION
    recall_w = cr['weighted avg']['recall']  # WEIGHTED-RECALL

    mcc = matthews_corrcoef(truth, prediction)
    cok = cohen_kappa_score(truth, prediction)

    # y_actu = pd.Series(truth, name='actual')
    # y_pred = pd.Series(prediction, name='predicted')
    mat_confusion = confusion_matrix(truth, prediction)

    val_dict = {
        cst.Metrics.F1.value:          float(f1score),
        cst.Metrics.F1_W.value:        float(f1score_w),
        cst.Metrics.PRECISION.value:   float(precision),
        cst.Metrics.PRECISION_W.value: float(precision_w),
        cst.Metrics.RECALL.value:      float(recall),
        cst.Metrics.RECALL_W.value:    float(recall_w),
        cst.Metrics.ACCURACY.value:    float(accuracy),
        cst.Metrics.MCC.value:         float(mcc),
        cst.Metrics.COK.value:         float(cok),
        cst.Metrics.LOSS.value:        float(np.sum(loss_vals)),
        cst.Metrics.CM.value:          mat_confusion.tolist()
    }
    return val_dict
