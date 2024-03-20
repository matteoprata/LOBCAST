import time

import pytorch_lightning as pl

import torch
import numpy as np
import torch.nn as nn
import src.constants as cst
from src.metrics.metrics_learning import compute_metrics


class LOBCAST_NNEngine(pl.LightningModule):
    def __init__(self, neural_architecture, loss_weights, hps, metrics_log, wandb_log):
        super().__init__()
        self.neural_architecture = neural_architecture
        self.loss_weights = loss_weights
        self.hps = hps
        self.metrics_log = metrics_log
        self.wandb_log = wandb_log

    def log_wandb(self, metrics):
        if self.wandb_log:
            self.wandb_log.log(metrics)

    def forward(self, batch):
        # time x features - 40 x 100 in general
        out = self.neural_architecture(batch)
        logits = nn.Softmax(dim=1)(out)  # todo check if within model
        return out, logits

    def training_step(self, batch, batch_idx):
        prediction_ind, y, loss_val, logits = self.make_predictions(batch)
        return {"loss": loss_val, "other": (prediction_ind, y, loss_val, logits)}

    def validation_step(self, batch, batch_idx):
        prediction_ind, y, loss_val, logits = self.make_predictions(batch)
        return prediction_ind, y, loss_val, logits

    def test_step(self, batch, batch_idx):
        prediction_ind, y, loss_val, logits = self.make_predictions(batch)
        return prediction_ind, y, loss_val, logits

    def make_predictions(self, batch):
        x, y = batch
        out, logits = self(x)
        loss_val = nn.CrossEntropyLoss(self.loss_weights)(out, y)

        # deriving prediction from softmax probs
        prediction_ind = torch.argmax(logits, dim=1)  # B
        return prediction_ind, y, loss_val, logits

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        t0 = time.time()
        self(x)
        torch.cuda.current_stream().synchronize()
        t1 = time.time()
        elapsed = t1 - t0
        print("Inference for the model:", elapsed, "ms")
        return elapsed

    def evaluate_classifier(self, stp_type, step_outputs):
        preds, truths, loss_vals, logits = self.__get_prediction_vectors(step_outputs)
        eval_dict = compute_metrics(truths, preds, loss_vals)

        var_name = "{}_{}".format(stp_type, cst.Metrics.LOSS.value)
        self.log(var_name, eval_dict[cst.Metrics.LOSS.value], prog_bar=True)

        var_name = "{}_{}".format(stp_type, cst.Metrics.F1.value)
        self.log(var_name, eval_dict[cst.Metrics.F1.value], prog_bar=True)

        path = cst.METRICS_BEST_FILE_NAME if self.metrics_log.is_best_model else cst.METRICS_RUNNING_FILE_NAME

        print("\n")
        print(f"END epoch {self.current_epoch} ({stp_type})")
        print("Logging stats...")
        self.metrics_log.add_metric(self.current_epoch, stp_type, eval_dict)
        self.metrics_log.dump_metrics(path)
        self.log_wandb({f"{stp_type}_{k}": v for k, v in eval_dict.items()})
        print("Done.")

    def training_epoch_end(self, training_step_outputs):
        training_step_outputs = [batch["other"] for batch in training_step_outputs]
        self.evaluate_classifier(cst.ModelSteps.TRAINING.value, training_step_outputs)

    def validation_epoch_end(self, validation_step_outputs):
        self.evaluate_classifier(cst.ModelSteps.VALIDATION.value, validation_step_outputs)

    def test_epoch_end(self, test_step_outputs):
        self.evaluate_classifier(cst.ModelSteps.TESTING.value, test_step_outputs)

    def __get_prediction_vectors(self, model_output):
        """ Accumulates the models output after each validation and testing epoch end. """

        preds, truths, losses, logits = [], [], [], []
        for preds_b, y_b, loss_val, logits_b in model_output:
            preds += preds_b.tolist()
            truths += y_b.tolist()
            logits += logits_b.tolist()
            losses += [loss_val.item()]  # loss is single per batch

        preds  = np.array(preds)
        truths = np.array(truths)
        logits = np.array(logits)
        losses = np.array(losses)

        return preds, truths, losses, logits

    def configure_optimizers(self):
        if self.hps.OPTIMIZER == "SGD":
            return torch.optim.SGD(self.parameters(), lr=self.hps.LEARNING_RATE)
        elif self.hps.OPTIMIZER == "ADAM":
            return torch.optim.Adam(self.parameters(), lr=self.hps.LEARNING_RATE)
        elif self.hps.OPTIMIZER == "RMSPROP":
            return torch.optim.RMSprop(self.parameters(), lr=self.hps.LEARNING_RATE)
