from pprint import pprint

import pytorch_lightning as pl

import torch
import torch.nn as nn

from sklearn.metrics import classification_report

import numpy as np
import src.config as co
import wandb

class NNEngine(pl.LightningModule):

    def __init__(
        self,
        model_type,
        neural_architecture,
        lr,
        remote_log=None
    ):

        super().__init__()

        self.remote_log = remote_log

        self.model_type = model_type
        self.neural_architecture = neural_architecture

        self.softmax = nn.Softmax(dim=1)

        self.loss_fn = nn.CrossEntropyLoss()

        self.lr = lr

    def forward(self, x):
        out = self.neural_architecture(x)
        logits = self.softmax(out)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        prediction = self(x)
        loss = self.loss_fn(prediction, y)
        return loss

    def validation_step(self, batch, batch_idx):
        prediction_ind, y, loss_val = self.__validation_and_testing(batch)
        return prediction_ind, y, loss_val

    def test_step(self, batch, batch_idx):
        prediction_ind, y, loss_val = self.__validation_and_testing(batch)
        return prediction_ind, y, loss_val

    def training_epoch_end(self, validation_step_outputs):
        losses = [el["loss"].item() for el in validation_step_outputs]
        sum_losses = float(np.sum(losses))
        self.log(co.ModelSteps.TRAINING.value + co.Metrics.LOSS.value, sum_losses, prog_bar=True)

        if self.remote_log is not None:
            self.remote_log.log({co.ModelSteps.TRAINING.value + co.Metrics.LOSS.value: sum_losses})

    def validation_epoch_end(self, validation_step_outputs):
        self.__validation_and_testing_end(validation_step_outputs, model_step=co.ModelSteps.VALIDATION)

    def test_epoch_end(self, validation_step_outputs):
        self.__validation_and_testing_end(validation_step_outputs, model_step=co.ModelSteps.TESTING)

    # COMMON
    def __validation_and_testing(self, batch):
        x, y = batch
        prediction = self(x)
        loss_val = self.loss_fn(prediction, y)

        # deriving prediction from softmax probs
        prediction_ind = torch.argmax(prediction, dim=1)

        return prediction_ind, y, loss_val

    def __validation_and_testing_end(self, validation_step_outputs, model_step):

        predictions, ys, loss_vals = [], [], []
        for prediction, y, loss_val in validation_step_outputs:
            predictions += prediction.tolist()
            ys += y.tolist()
            loss_vals += [loss_val.item()]

        cr = classification_report(ys, predictions, output_dict=True, zero_division=0)
        accuracy = cr['accuracy']  # MICRO-F1
        f1score = cr['macro avg']['f1-score'] #MACRO-F1
        precision = cr['macro avg']['precision'] #MACRO-PRECISION
        recall = cr['macro avg']['recall'] #MACRO-RECALL

        val_dict = {
            model_step.value + co.Metrics.LOSS.value:      float(np.sum(loss_vals)),
            model_step.value + co.Metrics.F1.value:        float(f1score),
            model_step.value + co.Metrics.PRECISION.value: float(precision),
            model_step.value + co.Metrics.RECALL.value:    float(recall),
            model_step.value + co.Metrics.ACCURACY.value:  float(accuracy)
        }

        # for saving best model
        self.log(model_step.value + co.Metrics.F1.value, f1score, prog_bar=True)

        if self.remote_log is not None:  # log to wandb
            self.remote_log.log(val_dict)
            self.remote_log.log({model_step.value + "_conf_mat": wandb.plot.confusion_matrix(
                probs=None,
                y_true=ys, preds=predictions,
                class_names=co.CLASS_NAMES,
                title=model_step.value + "_conf_mat")})

    def configure_optimizers(self):
        if self.model_type == co.Models.CNN2 or self.model_type == co.Models.CNNLSTM:
            return torch.optim.RMSprop(self.parameters(), lr=self.lr)
        else:
            return torch.optim.Adam(self.parameters(), lr=self.lr)