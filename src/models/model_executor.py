from pprint import pprint

import pandas as pd
import pytorch_lightning as pl

import torch
import torch.nn as nn

from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

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
        eps=None,
        weight_decay=0,
        loss_weights=None,
        remote_log=None,
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
        self.eps = eps

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
        preds, truths, loss_vals, stock_names = self.get_prediction_vectors(validation_step_outputs)

        model_step = cst.ModelSteps.VALIDATION

        # COMPUTE CM (1) (SRC) - (SRC)
        self.__log_wandb_cm(truths, preds, model_step, self.config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name)  # cm to log
        val_dict = self.__compute_metrics(truths, preds, model_step, loss_vals, self.config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name)  # dict to log

        # for saving best model
        validation_string = "{}_{}_{}".format(model_step.value, self.config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name, cst.Metrics.F1.value)
        self.log(validation_string, val_dict[validation_string], prog_bar=True)  # validation_!SRC!_F1

        if self.remote_log is not None:  # log to wandb
            self.remote_log.log(val_dict)

    def test_epoch_end(self, test_step_outputs):
        preds, truths, loss_vals, stock_names = self.get_prediction_vectors(test_step_outputs)

        model_step = cst.ModelSteps.TESTING

        # COMPUTE CM (1) (SRC) - (SRC)
        self.__log_wandb_cm(truths, preds, model_step, self.config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name)  # cm to log

        val_dict = self.__compute_metrics(truths, preds, model_step, loss_vals, self.config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name)  # dict to log
        self.config.METRICS_JSON.add_testing_metrics(self.config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name, val_dict)

        cm = self.__compute_sk_cm(truths, preds)
        self.config.METRICS_JSON.add_testing_cfm(self.config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name, cm)

        # PER STOCK PREDICTIONS
        if self.config.CHOSEN_STOCKS[cst.STK_OPEN.TEST] == cst.Stocks.ALL:
            # computing metrics per stock
            df = pd.DataFrame(
                list(zip(stock_names, preds, truths)),
                columns=['stock_names', 'predictions', 'ys']
            )

            for si in self.config.CHOSEN_STOCKS[cst.STK_OPEN.TEST].value:
                df_si = df[df['stock_names'] == si]
                truths_si = df_si['ys'].to_numpy()
                preds_si = df_si['predictions'].to_numpy()

                dic_si = self.__compute_metrics(truths_si, preds_si, model_step, loss_vals, si)
                self.config.METRICS_JSON.add_testing_metrics(si, dic_si)
                val_dict.update(dic_si)

                self.__log_wandb_cm(truths_si, preds_si, model_step, si)
                cm = self.__compute_sk_cm(truths_si, preds_si)
                self.config.METRICS_JSON.add_testing_cfm(si, cm)

        if self.remote_log is not None:  # log to wandb
            self.remote_log.log(val_dict)

    # COMMON
    def __validation_and_testing(self, batch):
        x, y, stock_names = batch
        prediction = self(x)
        loss_val = self.loss_fn(prediction, y)

        # deriving prediction from softmax probs
        prediction_ind = torch.argmax(prediction, dim=1)

        return prediction_ind, y, loss_val, stock_names

    def get_prediction_vectors(self, model_output):
        preds, truths, losses, stock_names = [], [], [], []
        for preds_b, y_b, loss_val, stock_name_b in model_output:
            preds += preds_b.tolist()
            truths += y_b.tolist()
            stock_names += stock_name_b
            # loss is single per batch
            losses += [loss_val.item()]

        preds = np.array(preds)
        truths = np.array(truths)
        losses = np.array(losses)

        if self.config.CHOSEN_MODEL == cst.Models.DEEPLOBATT:
            truths = np.argmax(truths, axis=1)
            truths = truths[:, cst.FI_HORIZONS_MAPPINGS[self.config.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON]]]
            preds = preds[:, cst.FI_HORIZONS_MAPPINGS[self.config.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON]]]

        return preds, truths, losses, stock_names

    def __log_wandb_cm(self, ys, predictions, model_step, si):
        if self.remote_log is not None:  # log to wandb
            name = model_step.value + f"_conf_mat_{si}"
            self.remote_log.log({name: wandb.plot.confusion_matrix(
                probs=None,
                y_true=ys, preds=predictions,
                class_names=[cl.name for cl in cst.Predictions],
                title=name)},
            )

    def __compute_sk_cm(self, truths, predictions):
        y_actu = pd.Series(truths, name='actual')
        y_pred = pd.Series(predictions, name='predicted')
        mat_confusion = confusion_matrix(y_actu, y_pred)
        return mat_confusion

    def __compute_metrics(self, ys, predictions, model_step, loss_vals, si):
        ys = torch.Tensor(ys)
        predictions = torch.Tensor(predictions)

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
            return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=self.eps)
        elif self.optimizer == cst.Optimizers.RMSPROP.value:
            return torch.optim.RMSprop(self.parameters(), lr=self.lr)
