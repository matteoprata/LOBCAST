from pprint import pprint

import pandas as pd
import pytorch_lightning as pl

import torch
import torch.nn as nn

from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef


import numpy as np
import src.constants as cst
import wandb
from src.config import Configuration


class NNEngine(pl.LightningModule):

    def __init__(
        self,
        config: Configuration,
        model_type,
        neural_architecture,
        optimizer,
        lr,
        weight_decay=0,
        loss_weights=None,
        remote_log=None
    ):
        super().__init__()
        assert optimizer == cst.Optimizers.ADAM.value or optimizer == cst.Optimizers.RMSPROP.value

        self.config = config
        self.remote_log = remote_log

        self.model_type = model_type
        self.neural_architecture = neural_architecture

        self.softmax = nn.Softmax(dim=1)

        self.loss_fn = nn.CrossEntropyLoss(weight=loss_weights)

        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        out = self.neural_architecture(x)
        logits = self.softmax(out)
        return logits

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        prediction = self(x)
        loss = self.loss_fn(prediction, y)
        return loss

    def validation_step(self, batch, batch_idx):
        prediction_ind, y, loss_val, stock_names = self.__validation_and_testing(batch)
        return prediction_ind, y, loss_val, stock_names

    def test_step(self, batch, batch_idx):
        prediction_ind, y, loss_val, stock_names = self.__validation_and_testing(batch)
        return prediction_ind, y, loss_val, stock_names

    def training_epoch_end(self, validation_step_outputs):
        losses = [el["loss"].item() for el in validation_step_outputs]
        sum_losses = float(np.sum(losses))
        var_name = cst.ModelSteps.TRAINING.value + cst.Metrics.LOSS.value
        self.log(var_name, sum_losses, prog_bar=True)

        if self.remote_log is not None:
            self.remote_log.log({var_name: sum_losses})

    def validation_epoch_end(self, validation_step_outputs):
        self.__validation_and_testing_end(validation_step_outputs, model_step=cst.ModelSteps.VALIDATION)

    def test_epoch_end(self, test_step_outputs):
        self.__validation_and_testing_end(test_step_outputs, model_step=cst.ModelSteps.TESTING)

    # COMMON
    def __validation_and_testing(self, batch):
        x, y, stock_names = batch
        prediction = self(x)
        loss_val = self.loss_fn(prediction, y)

        # deriving prediction from softmax probs
        prediction_ind = torch.argmax(prediction, dim=1)

        return prediction_ind, y, loss_val, stock_names

    def __validation_and_testing_end(self, validation_step_outputs, model_step):

        predictions, ys, loss_vals, stock_names = [], [], [], []
        for predictions_b, y_b, loss_val, stock_name_b in validation_step_outputs:
            predictions += predictions_b.tolist()
            ys += y_b.tolist()
            stock_names += stock_name_b
            # loss is single per batch
            loss_vals += [loss_val.item()]

        self.__compute_cm(ys, predictions, model_step, self.config.SRC_STOCK_NAME)                             # cm to log
        val_dict = self.__compute_metrics(ys, predictions, model_step, loss_vals, self.config.SRC_STOCK_NAME)  # dict to log

        if model_step == cst.ModelSteps.TESTING and self.config.CHOSEN_STOCKS[cst.STK_OPEN.TEST] == cst.Stocks.ALL:
            # computing metrics per stock
            df = pd.DataFrame(
                list(zip(stock_names, predictions, ys)),
                columns=['stock_names', 'predictions', 'ys']
            )

            for si in self.config.CHOSEN_STOCKS[cst.STK_OPEN.TEST].value:
                df_si = df[df['stock_names'] == si]
                ys = df_si['ys'].to_numpy()
                predictions = df_si['predictions'].to_numpy()
                val_dict.update(self.__compute_metrics(ys, predictions, model_step, loss_vals, si))

                self.__compute_cm(ys, predictions, model_step, si)

        # for saving best model
        validation_string = model_step.value + "_{}_".format(self.config.SRC_STOCK_NAME) + cst.Metrics.F1.value
        self.log(validation_string, val_dict[validation_string], prog_bar=True)   # validation_!SRC!_F1

        if self.remote_log is not None:  # log to wandb
            self.remote_log.log(val_dict)

    def __compute_cm(self, ys, predictions, model_step, si):
        if self.remote_log is not None:  # log to wandb
            name = model_step.value + f"_conf_mat_{si}"
            self.remote_log.log({name: wandb.plot.confusion_matrix(
                probs=None,
                y_true=ys, preds=predictions,
                class_names=self.config.CLASS_NAMES,
                title=name)}
            )

    def __compute_metrics(self, ys, predictions, model_step, loss_vals, si):

        cr = classification_report(ys, predictions, output_dict=True, zero_division=0)
        accuracy = cr['accuracy']  # MICRO-F1
        f1score = cr['macro avg']['f1-score']  # MACRO-F1
        precision = cr['macro avg']['precision']  # MACRO-PRECISION
        recall = cr['macro avg']['recall']  # MACRO-RECALL

        mcc = matthews_corrcoef(ys, predictions)

        val_dict = {
            model_step.value + f"_{si}_" + cst.Metrics.F1.value: float(f1score),
            model_step.value + f"_{si}_" + cst.Metrics.PRECISION.value: float(precision),
            model_step.value + f"_{si}_" + cst.Metrics.RECALL.value: float(recall),
            model_step.value + f"_{si}_" + cst.Metrics.ACCURACY.value: float(accuracy),
            model_step.value + f"_{si}_" + cst.Metrics.MCC.value: float(mcc),
            # single
            model_step.value + f"__" + cst.Metrics.LOSS.value: float(np.sum(loss_vals)),
        }

        return val_dict

    def configure_optimizers(self):

        if self.model_type == cst.Models.DAIN:
            return torch.optim.RMSprop([
                {'params': self.neural_architecture.base.parameters()},
                {'params': self.neural_architecture.dean.mean_layer.parameters(), 'lr': self.lr*self.neural_architecture.dean.mean_lr},
                {'params': self.neural_architecture.dean.scaling_layer.parameters(), 'lr': self.lr*self.neural_architecture.dean.scale_lr},
                {'params': self.neural_architecture.dean.gating_layer.parameters(), 'lr': self.lr*self.neural_architecture.dean.gate_lr},
            ], lr=self.lr)

        if self.optimizer == cst.Optimizers.ADAM.value:
            return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == cst.Optimizers.RMSPROP.value:
            return torch.optim.RMSprop(self.parameters(), lr=self.lr)
