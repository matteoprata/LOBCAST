import time

import pandas as pd
import pytorch_lightning as pl

import torch
import torch.nn as nn

import numpy as np
import src.constants as cst
import wandb
from src.config import Configuration
from src.utils.utilities import get_index_from_window
from src.metrics.metrics_learning import compute_sk_cm, compute_metrics


class NNEngine(pl.LightningModule):

    def __init__(
        self,
        config: Configuration,
        model_type=None,
        neural_architecture=None,
        optimizer=None,
        lr=None,
        eps=None,
        weight_decay=0,
        momentum=None,
        loss_weights=None,
        remote_log=None,
        n_samples=None,
        n_epochs=None,
        n_batch_size=None,
    ):
        super().__init__()

        self.config = config

        self.n_samples = n_samples
        self.n_epochs = n_epochs
        self.n_batch_size = n_batch_size

        self.remote_log = remote_log

        self.model_type = model_type
        self.neural_architecture = neural_architecture

        self.softmax = nn.Softmax(dim=1)

        self.loss_fn = nn.CrossEntropyLoss(weight=loss_weights)

        self.optimizer_name = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.eps = eps
        self.momentum = momentum

        self.testing_mode = cst.ModelSteps.TESTING

        # LR decay here below
        self.optimizer_obj = None
        self.no_improvement_count = 0
        self.best_epoch_train_loss = 0
        self.cur_decay_index = 0
        self.LR_DECAY_CTABL = [0.005, 0.001, 0.0005, 0.0001, 0.00008, 0.00001]

    def forward(self, x):
        out = self.neural_architecture(x)
        logits = self.softmax(out)

        return out, logits

    def training_step(self, batch, batch_idx):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch

        out, logits = self(x)

        loss = self.loss_fn(out, y)
        return loss

    def validation_step(self, batch, batch_idx):
        prediction_ind, y, loss_val, stock_names, logits = self.__validation_and_testing(batch)
        return prediction_ind, y, loss_val, stock_names, logits

    def test_step(self, batch, batch_idx):
        prediction_ind, y, loss_val, stock_names, logits = self.__validation_and_testing(batch)
        return prediction_ind, y, loss_val, stock_names, logits

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if len(batch) == 3:
            x, _, _ = batch
        else:
            x, _ = batch

        t0 = time.time()
        p = self(x)
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
        elapsed = t1 - t0
        # print("Inference for the model:", elapsed, "ms")
        return elapsed

    def training_epoch_end(self, validation_step_outputs):
        losses = [el["loss"].item() for el in validation_step_outputs]
        sum_losses = float(np.sum(losses))

        self.update_lr(sum_losses)

        var_name = cst.ModelSteps.TRAINING.value + cst.Metrics.LOSS.value
        self.log(var_name, sum_losses, prog_bar=True)

        if self.remote_log is not None:
            self.remote_log.log({var_name: sum_losses})

        # self.config.METRICS_JSON.add_testing_metrics(self.config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name, {'MAX-EPOCHS': self.current_epoch})
        # self.remote_log({"current_epoch": self.current_epoch})

    def validation_epoch_end(self, validation_step_outputs):
        preds, truths, loss_vals, stock_names, logits = self.get_prediction_vectors(validation_step_outputs)

        model_step = cst.ModelSteps.VALIDATION_EPOCH

        # COMPUTE CM (1) (SRC) - (SRC)
        self.__log_wandb_cm(truths, preds, model_step, self.config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name)  # cm to log
        val_dict = compute_metrics(truths, preds, model_step, loss_vals, self.config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name)  # dict to log

        # for saving best model
        validation_string = "{}_{}_{}".format(model_step.value, self.config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name, cst.Metrics.F1.value)
        self.log(validation_string, val_dict[validation_string], prog_bar=True)  # validation_!SRC!_F1

        if self.remote_log is not None:  # log to wandb
            self.remote_log.log(val_dict)

    def test_epoch_end(self, test_step_outputs):
        preds, truths, loss_vals, stock_names, logits = self.get_prediction_vectors(test_step_outputs)

        model_step = self.testing_mode

        # LOGGING
        # COMPUTE CM (1) (SRC) - (SRC)
        self.__log_wandb_cm(truths, preds, model_step, self.config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name)  # cm to log

        val_dict = compute_metrics(truths, preds, model_step, loss_vals, self.config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name)  # dict to log
        self.config.METRICS_JSON.update_metrics(self.config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name, val_dict)

        logits_dict = {'LOGITS': str(logits.tolist())}
        self.config.METRICS_JSON.update_metrics(self.config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name, logits_dict)

        cm = compute_sk_cm(truths, preds)
        self.config.METRICS_JSON.update_cfm(self.config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name, cm)

        # PER STOCK PREDICTIONS
        if self.config.CHOSEN_STOCKS[cst.STK_OPEN.TEST] == cst.Stocks.ALL and self.config.CHOSEN_MODEL not in [cst.Models.METALOB, cst.Models.MAJORITY]:
            # computing metrics per stock

            for si in self.config.CHOSEN_STOCKS[cst.STK_OPEN.TEST].value:
                index_si = np.where(stock_names == si)[0]
                truths_si = truths[index_si]
                preds_si = preds[index_si]
                logits_si = logits[index_si]

                dic_si = compute_metrics(truths_si, preds_si, model_step, loss_vals, si)
                self.config.METRICS_JSON.update_metrics(si, dic_si)
                val_dict.update(dic_si)

                logits_dict = {'LOGITS': str(logits_si.tolist())}
                self.config.METRICS_JSON.update_metrics(si, logits_dict)

                self.__log_wandb_cm(truths_si, preds_si, model_step, si)

                cm = compute_sk_cm(truths_si, preds_si)
                self.config.METRICS_JSON.update_cfm(si, cm)

        if self.remote_log is not None:  # log to wandb
            val_dict["current_epoch"] = self.current_epoch
            self.remote_log.log(val_dict)

    # COMMON
    def __validation_and_testing(self, batch):

        stock_names = [None]*self.n_batch_size
        if len(batch) == 3:
            x, y, stock_names = batch
        else:
            x, y = batch

        out, logits = self(x)       # B x 3;   B X 3

        loss_val = self.loss_fn(out, y)

        # deriving prediction from softmax probs
        prediction_ind = torch.argmax(logits, dim=1)  # B

        return prediction_ind, y, loss_val, stock_names, logits

    def get_prediction_vectors(self, model_output):
        """ Accumulates the models output after each validation and testing epoch end. """

        preds, truths, losses, stock_names, logits = [], [], [], [], []
        for preds_b, y_b, loss_val, stock_name_b, logits_b in model_output:
            preds += preds_b.tolist()
            truths += y_b.tolist()
            logits += logits_b.tolist()
            stock_names += stock_name_b
            # loss is single per batch
            losses += [loss_val.item()]

        preds = np.array(preds)
        truths = np.array(truths)
        logits = np.array(logits)
        stock_names = np.array(stock_names)
        losses = np.array(losses)

        if self.config.CHOSEN_MODEL == cst.Models.DEEPLOBATT:
            index = cst.HORIZONS_MAPPINGS_FI[self.config.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW]]
            truths = truths[:, index]
            preds = preds[:, index]

        return preds, truths, losses, stock_names, logits

    def __log_wandb_cm(self, ys, predictions, model_step, si):
        if self.remote_log is not None:  # log to wandb
            name = model_step.value + f"_conf_mat_{si}"
            self.remote_log.log({name: wandb.plot.confusion_matrix(
                probs=None,
                y_true=ys, preds=predictions,
                class_names=[cl.name for cl in cst.Predictions],
                title=name)},
            )

    def configure_optimizers(self):

        if self.optimizer_name == cst.Optimizers.ADAM.value:

            if self.config.CHOSEN_MODEL == cst.Models.ATNBoF:
                opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int((self.n_samples / self.n_batch_size) * self.n_epochs))
                self.optimizer_obj = opt
                return [opt], [{"scheduler": sch, "interval": "step", "frequency": 1}]
            else:
                self.optimizer_obj = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, eps=self.eps)
                return self.optimizer_obj

        elif self.optimizer_name == cst.Optimizers.RMSPROP.value:
            if self.model_type == cst.Models.DAIN:
                self.optimizer_obj = torch.optim.RMSprop([
                    {'params': self.neural_architecture.base.parameters()},
                    {'params': self.neural_architecture.dean.mean_layer.parameters(),
                     'lr': self.lr * self.neural_architecture.dean.mean_lr},
                    {'params': self.neural_architecture.dean.scaling_layer.parameters(),
                     'lr': self.lr * self.neural_architecture.dean.scale_lr},
                    {'params': self.neural_architecture.dean.gating_layer.parameters(),
                     'lr': self.lr * self.neural_architecture.dean.gate_lr},
                ], lr=self.lr)
                return self.optimizer_obj
            else:
                self.optimizer_obj = torch.optim.RMSprop(self.parameters(), lr=self.lr)
                return self.optimizer_obj

        elif self.optimizer_name == cst.Optimizers.SGD.value:
            if self.config.CHOSEN_MODEL == cst.Models.AXIALLOB or self.config.CHOSEN_MODEL == cst.Models.METALOB:
                opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int((self.n_samples / self.n_batch_size) * self.n_epochs))
                self.optimizer_obj = opt
                return [opt], [{"scheduler": sch, "interval": "step", "frequency": 1}]

        # # CHECK FOR ALL
        # if self.current_epoch == 0:
        #     print("check")
        #     for g in self.optimizer_obj.param_groups:
        #         print(g['lr'])
        #     print("OK")

    def update_lr(self, sum_loss):
        """ To run on train epoch end, to update the lr of needing models. """

        if self.config.CHOSEN_MODEL == cst.Models.ATNBoF:
            DROP_EPOCHS = [11, 51]
            if self.current_epoch in DROP_EPOCHS:
                self.lr /= 0.1
                self.__update_all_lr()

        elif self.config.CHOSEN_MODEL == cst.Models.CTABL:

            if sum_loss > self.best_epoch_train_loss and self.current_epoch > 0:
                self.no_improvement_count += 1
            else:
                self.best_epoch_train_loss = sum_loss
                self.no_improvement_count = 0

            if self.no_improvement_count > 3 and self.cur_decay_index < len(self.LR_DECAY_CTABL):
                self.no_improvement_count = 0
                self.lr = self.LR_DECAY_CTABL[self.cur_decay_index]
                self.cur_decay_index += 1

                self.__update_all_lr()

        elif self.config.CHOSEN_MODEL == cst.Models.BINCTABL:
            DROP_EPOCHS = [11, 71]
            if self.current_epoch == DROP_EPOCHS[0]:
                self.lr = 1e-4
                self.__update_all_lr()

            elif self.current_epoch == DROP_EPOCHS[1]:
                self.lr = 1e-5
                self.__update_all_lr()

    def __update_all_lr(self):
        # SETTING the new LR
        for g in self.optimizer_obj.param_groups:
            g['lr'] = self.lr
