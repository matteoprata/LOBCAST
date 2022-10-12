import json

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch

import sys
import os
sys.path.append(os.getcwd())  # to work with WANDB

from src import config as conf
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, random_split
from src.generator.baseapproach import PollutionType
from src.utils.lobdataset import RawDataset, LOBDataset, DEEPDataset
from src.costpredictor import markovian_cost_lobster, markovian_cost_fi
from src.models.cnn1 import CNN
from src.models.mlp import MLP
from src.models.cnn1 import CNN
from src.models.cnn2 import CNN2
from src.models.lstm import LSTM
from src.models.deeplob import DeepLob
from src.generator import gan_utils

from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn import metrics

import wandb
import os
from collections import OrderedDict, defaultdict

PATH_DATASETS = os.environ.get("PATH_DATASETS", "")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 32 if AVAIL_GPUS else 32  # TODO: sistemare parametri da command line
NUM_WORKERS = int(os.cpu_count() / 2)
POLLUTION = PollutionType.BUY_ONLY
N_LEVELS = 10
NUM_COL = 10 - 1
DEVICE = torch.device(conf.DEVICE)
WANDB_IN = None
lob_data = None
ths = None
# https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/basic-gan.html
MIN_VOLUME = None


class LOBDataModule(LightningDataModule):
    def __init__(
            self,
            lobdaset: RawDataset,
            window_size: int,
            horizon: int,
            batch_size: int = BATCH_SIZE,
            num_workers: int = NUM_WORKERS,
    ):
        super().__init__()
        self.window_size = window_size
        self.horizon = horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lobdaset = lobdaset

        # Setting default dims here because we know them.
        # Could optionally be assigned dynamically in dm.setup()

        self.dims = (self.window_size, 40)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.d_train = self.lobdaset.split_train_data(torch_dataset=True)
            self.d_val = self.lobdaset.split_val_data(torch_dataset=True)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.d_test = self.lobdaset.split_test_data(torch_dataset=True)

    def train_dataloader(self):
        return DataLoader(self.d_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.d_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.d_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=True)


# # # # # # # # # # # GENERATOR # # # # # # # # # # #

class Generator(nn.Module):
    def __init__(self, latent_dim, mask_shape):
        super().__init__()
        self.mask_shape = mask_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),         # GAUSSIAN RANDOMNESS
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.mask_shape))),  # nx9
            nn.Tanh(),                                       # [-1,1]
        )

    def forward(self, z):
        pert = self.model(z)
        pert = pert.view(BATCH_SIZE, *self.mask_shape)  # torch.reshape(pert, (BATCH_SIZE, *self.mask_shape))
        return pert

# # # # # # # # # # # DISCRIMINATOR # # # # # # # # # # #


class Discriminator(nn.Module):
    def __init__(self, lob_slice_shape):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(lob_slice_shape)), 512),   # b x n x 40
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),                               # [0,1] \in R
            nn.Sigmoid(),
        )

    def forward(self, lob_slice):
        lob_slice_flat = lob_slice.view(lob_slice.size(0), -1)
        validity = self.model(lob_slice_flat)

        return validity


# # # # # # # # # # # GAN # # # # # # # # # # #

class GAN(LightningModule):
    def __init__(
            self,
            # channels,
            rows,
            columns,
            model,
            horizon,
            costpredictor_columns,
            latent_dim: int = 100,
            lr: float = 0.0002,
            b1: float = 0.5,
            b2: float = 0.999,
            batch_size: int = BATCH_SIZE,
            min_volume:int=MIN_VOLUME,
            stats_final_dict=None,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.predictor_model = load_model(model, horizon=horizon)
        # this model weights should not be trained
        for param in self.predictor_model.parameters():
            param.requires_grad = False

        self.stats_final_dict = stats_final_dict
        self.trend_criterion = nn.CrossEntropyLoss()
        self.costpredictor_columns = costpredictor_columns

        # target class (-1=0, 0=1, 1=2)
        #
        # ratio: placing orders (volume 1) on the sell side will decrease the volume, thus classifiers may be oriented
        # -----| to predict an upward trend (many want to buy), we force the classifier to predict a downward trend
        # -----| anyways. To see if everything works out good.
        # -----|
        self.target = torch.zeros(batch_size, dtype=torch.long).to(device=conf.DEVICE)

        # networks
        lob_data_shape = (rows, columns)
        self.mask_data_shape = (rows, NUM_COL)

        self.generator = Generator(latent_dim=self.hparams.latent_dim, mask_shape=self.mask_data_shape)
        self.discriminator = Discriminator(lob_slice_shape=lob_data_shape)

        self.loss_log = defaultdict(list)
        self.is_model_end_validating = False

        self.weights_losses = WEIGHT_LOSS  # loss costs
        self.min_volume = min_volume

        self.is_first_validation = True
        self.no_perturbation_predictions = dict()

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):             # [0,1,1,1,0] [1,1,1,1,1]
        return F.binary_cross_entropy(y_hat, y)

    def trend_loss(self, y_hat, y):             # [12,1,1,1,0] [13,1,1,1,1]
        y = y.to(torch.long)
        loss = self.trend_criterion(y_hat, y)
        return loss 

    def training_step(self, batch, batch_idx, optimizer_idx):

        if self.costpredictor_columns:
            pre_X, truth = batch
            lob_orders = pre_X[..., :-11]   # B,?,10x40
            volumes = pre_X[..., -11:-1]    # 10x10   # TODO enlarge to 20 (BUY / SELL)
            volatilities = pre_X[..., -1:]  # 10x1

            lob_orders = torch.squeeze(lob_orders)
            volumes = torch.squeeze(volumes)
            volatilities = torch.squeeze(volatilities)
            volatilities = volatilities.reshape(BATCH_SIZE, volatilities.shape[-1], 1)
            S = get_intralevel_midprice(lob_orders)
        else:
            lob_orders, truth = batch         # ([X SLICE],[Y])
            # remove a dimension that we don't need
            lob_orders = torch.squeeze(lob_orders)

        # lob_orders, truth, outstanding_volume, volatilty
        lob_orders = lob_orders.float()

        # sample noise
        z = torch.randn(BATCH_SIZE, self.hparams.latent_dim)  # lob_orders.shape[0] == batch
        z = z.type_as(lob_orders)

        # train generator
        if optimizer_idx == 0:
            # generate perturbation and merge to slices
            perturbation = self(z)  # pert h x 9 -> h x (n_lev - 1)

            HEURISTIC_SCALERS = {"generator_loss": 100, "perturbation_cost": 2.5, "attacked_model_loss": 1.5}

            # >>> LOSS ELEMENT: cost
            perturbation_cost = compute_cost(perturbation, S, volatilities, volumes) / HEURISTIC_SCALERS["perturbation_cost"]
            # print("cost", perturbation_cost)

            perturbed_slices = gan_utils.add_perturbation(lob_orders, perturbation, min_volume=self.min_volume)
            perturbed_slices = perturbed_slices.to(device=conf.DEVICE)

            discriminator_out = self.discriminator(perturbed_slices)  # discriminator guess on perturbed instance
            # statement: TRUE (view of the generator)
            valid = torch.ones(BATCH_SIZE, 1)
            valid = valid.type_as(lob_orders)
            # >>> LOSS ELEMENT: perturbation realism
            generator_loss = self.adversarial_loss(discriminator_out, valid) / HEURISTIC_SCALERS["generator_loss"]  # 0,0,0,0 || 1,1,1,1 bad for generator
            generator_loss.requires_grad_(True)
            # print("perturbation realism", g_loss)

            in_perturbed_slices = torch.unsqueeze(perturbed_slices, 1)
            predicted_y = self.predictor_model(in_perturbed_slices)
            # >>> LOSS ELEMENT: classificator prediction
            attacked_model_loss = self.trend_loss(predicted_y, self.target) / HEURISTIC_SCALERS["attacked_model_loss"]
            # print("classificator prediction", f_loss)

            tot_loss = (+ attacked_model_loss * self.weights_losses["class"]
                        + generator_loss * self.weights_losses["gan"]
                        + perturbation_cost * self.weights_losses["cost"])

            self.loss_log["f_loss"].append(attacked_model_loss)
            self.loss_log["g_loss"].append(generator_loss)
            self.loss_log["perturbation_cost"].append(perturbation_cost)
            self.loss_log["tot_loss"].append(tot_loss)

            tqdm_dict = {"tot_loss": tot_loss}
            output = OrderedDict({"loss": tot_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # PART 1
            # how well can it label as real? statement: TRUE
            valid = torch.ones(lob_orders.size(0), 1)
            valid = valid.type_as(lob_orders)

            discou = self.discriminator(lob_orders)
            real_loss = self.adversarial_loss(discou, valid)

            # PART 2
            # how well can it label as fake? statement: FALSE
            fake = torch.zeros(BATCH_SIZE, 1)
            fake = fake.type_as(lob_orders)
            perturbation = self(z)

            perturbed_slices = gan_utils.add_perturbation(lob_orders, perturbation, min_volume=self.min_volume)

            pbs = self.discriminator(perturbed_slices)
            fake_loss = self.adversarial_loss(pbs, fake)  # 1,1,1,1 || 0,0,0,0
            fake_loss.requires_grad_(True)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2

            self.loss_log["real_loss"].append(real_loss)
            self.loss_log["fake_loss"].append(fake_loss)
            self.loss_log["d_loss"].append(d_loss)

            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

    def validation_step(self, batch, batch_idx):
        pre_X, truth = batch
        lob_orders = pre_X[..., :-11]  # B,?,10x40
        lob_orders = torch.squeeze(lob_orders)
        lob_orders = lob_orders.float()

        # sample noise
        z = torch.randn(BATCH_SIZE, self.hparams.latent_dim)  # lob_orders.shape[0] == batch
        z = z.type_as(lob_orders)

        perturbation = self(z)  # pert h x 9 -> h x (n_lev - 1)

        perturbed_slices = gan_utils.add_perturbation(lob_orders, perturbation, min_volume=self.min_volume)
        perturbed_slices = perturbed_slices.to(device=conf.DEVICE)

        in_perturbed_slices = torch.unsqueeze(perturbed_slices, 1)
        predicted_y = self.predictor_model(in_perturbed_slices)  # TODO CHECK WHY NO PROBS?
        f_loss = self.trend_loss(predicted_y, truth).item()

        predicted_y = torch.argmax(predicted_y, dim=1)

        # returns the validation without perturbation only once
        if self.is_first_validation:
            in_lob_orders = torch.unsqueeze(lob_orders, 1)
            prediction_vector = self.predictor_model(in_lob_orders)
            predicted_y_no_pert = torch.argmax(prediction_vector, dim=1)
            self.no_perturbation_predictions[batch_idx] = predicted_y_no_pert
        else:
            predicted_y_no_pert = self.no_perturbation_predictions[batch_idx]

        return f_loss, predicted_y_no_pert, predicted_y, truth

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def validation_epoch_end(self, validation_step_outputs):
        """ Suns at the end of validation epoch"""

        # merge all the stats
        losses, preds_no_per, preds_per, truths = [], [], [], []
        for lo, pred_no_per, pred_per, tru in validation_step_outputs:
            losses.append(lo)
            preds_no_per += [el.item() for el in pred_no_per]
            preds_per += [el.item() for el in pred_per]
            truths += [el.item() for el in tru]

        confusion_mat_no_per = metrics.confusion_matrix(preds_no_per, truths)
        confusion_mat_per = metrics.confusion_matrix(preds_per, truths)

        print("\nVAL confusion matrix no pert {} \n".format(self.current_epoch), confusion_mat_no_per)
        print("\nVAL confusion matrix pert {} \n".format(self.current_epoch), confusion_mat_per)

        precision, recall, fscore, _ = prfs(preds_per, truths, average="macro")

        val_dict = {'val_f_avgloss': float(mean(losses)),
                    'val_f1': float(fscore),
                    'val_prec': float(precision),
                    'val_recall': float(recall)
                    }

        confusion_mat_per_preds = confusion_mat_per.sum(axis=1)
        confusion_mat_no_per_preds = confusion_mat_no_per.sum(axis=1)

        # classes of the function with the epochs, number of predictions
        val_dict['class_-1_pert'] = int(confusion_mat_per_preds[0])
        val_dict['class_0_pert'] = int(confusion_mat_per_preds[1])
        val_dict['class_1_pert'] = int(confusion_mat_per_preds[2])

        # relative increase in instances
        val_dict['class_-1_rel'] = int(confusion_mat_per_preds[0]) / int(confusion_mat_no_per_preds[0]) - 1
        val_dict['class_0_rel'] =  int(confusion_mat_per_preds[1]) / int(confusion_mat_no_per_preds[1]) - 1
        val_dict['class_1_rel'] =  int(confusion_mat_per_preds[2]) / int(confusion_mat_no_per_preds[2]) - 1

        WANDB_IN.log(val_dict)

        # save stats in a json file
        for k, v in val_dict.items():
            self.stats_final_dict[k].append(v)

        # log a confusion matrix, as a list of lists
        confusion_mat_per_list = [int(val) for arr in confusion_mat_per for val in arr]
        confusion_mat_no_per_list = [int(val) for arr in confusion_mat_no_per for val in arr]
        self.stats_final_dict["confusion-pert"].append(confusion_mat_per_list)
        self.stats_final_dict["confusion-no-pert"].append(confusion_mat_no_per_list)

        self.is_model_end_validating = True
        self.is_first_validation = False

    def on_epoch_end(self):
        if self.is_model_end_validating:
            self.is_model_end_validating = False
            return

        if self.current_epoch > 0 and self.current_epoch % conf.SAVE_GAN_MODEL_EVERY == 0:
            torch.save(self.generator.state_dict(), conf.MODEL_GAN+"model_{}.mod".format(self.current_epoch))

        loss_log = {'gen_total_loss': float(mean(self.loss_log["tot_loss"])),
                    'gen_pred_loss':  float(mean(self.loss_log["f_loss"])),
                    'gen_cost_loss':  float(mean(self.loss_log["perturbation_cost"])),
                    'gen_dis_loss':   float(mean(self.loss_log["g_loss"])),
                    'dis_total_loss': float(mean(self.loss_log["d_loss"])),
                    'dis_real_loss':  float(mean(self.loss_log["real_loss"])),
                    'dis_fake_loss':  float(mean(self.loss_log["fake_loss"]))}

        WANDB_IN.log(loss_log)

        # save stats in a json file
        for k, v in loss_log.items():
            self.stats_final_dict[k].append(v)

        self.loss_log = defaultdict(list)  # reset the dict

        log_stats(stats_fname, self.stats_final_dict)


def mean(vec):
    if len(vec) == 0:
        return np.nan 
    else:
        return sum(vec)/len(vec)

# END GAN


def get_intralevel_midprice(M):
    """ Given market observation, it outputs a intralevel midprice matrix."""
    # psell, vsell, pbuy, vbuy

    # 100 | 99 | 98 | 97
    #      100 | 99 | 98 | 97
    # x     199 s2  s3
    # x     99.5 98.5
    # 99.5 98.5 ...

    initial_ind = None
    if POLLUTION == PollutionType.BUY_ONLY:     # dalla 2 ogni 4
        initial_ind = 2
    elif POLLUTION == PollutionType.SELL_ONLY:  # dalle 0 ogni 4
        initial_ind = 0

    dim_roll = range(len(M.shape))[-1]
    out = M[:, :, initial_ind::4]
    out_s = torch.roll(out, shifts=1, dims=dim_roll)
    # first column has no meaning
    out = (out + out_s) / 2
    return out


def compute_cost(A, S, V, M):
    """  This function allows to compute the adversarial cost in a probabilistic way.

        A: is the perturbation matrix hxL
        S: is the intralevel midprice matrix hxL
        V: is the volatilities vector hx1
        M: is the mass vector hxL
    """
    MA = M[:, :, 1:] * (A > 0)
    real_prices = S[:, :, 1:] * (A > 0)
    VO = V[:, :, :] * (A > 0)

    MA[MA == 0] = -1
    VO[VO == 0] = -1

    VO_th = ths["volatilies_quantiles.npy"]
    MA_th = ths["volume_quantiles.npy"]
    P_LAMB = ths["prob_exec.npy"]

    # VOLATILITY ARE ON THE COLUMNS (X AXIS)
    CON_W = [(VO != -1) & (VO < VO_th[0]),
             (VO != -1) & (VO > VO_th[0]) & (VO < VO_th[1]),
             (VO != -1) & (VO > VO_th[1]) & (VO < VO_th[2]),
             (VO != -1) & (VO > VO_th[2])]

    # VOLUMES ARE ON THE ROWS (Y AXIS)
    CON_O = [(MA != -1) & (MA < MA_th[0]),
             (MA != -1) & (MA > MA_th[0]) & (MA < MA_th[1]),
             (MA != -1) & (MA > MA_th[1]) & (MA < MA_th[2]),
             (MA != -1) & (MA > MA_th[2])]

    N_RANGES = 4
    PROBS = torch.zeros(size=MA.shape, device=DEVICE)
    for u in range(N_RANGES):
        for v in range(N_RANGES):
            PROBS += CON_W[u] * CON_O[v] * P_LAMB[u][v]

    TOTAL_EXPECTED_COST = float(torch.sum(PROBS * real_prices))  # no need to add A because there are -1
    return TOTAL_EXPECTED_COST

# noinspection PyPackageRequirements
def load_model(name, horizon):

    n_feat = 40
    n_classes = 3

    if name == "MLP":
        pre_model = MLP(n_feat * horizon, n_classes)
    elif name == "LSTM":
        pre_model = LSTM(n_classes, n_feat, 32, 1, horizon)
    elif name == "DeepLob":
        pre_model = DeepLob(n_classes)
    elif name == 'CNN':
        temp = 26 # horizon = 100
        if horizon == 10:
            temp = 3
        if horizon == 20:
            temp = 6
        elif horizon == 50:
            temp = 13
        pre_model = CNN(horizon, n_feat, n_classes, temp)
    elif name == "CNN2":
        temp = 249
        if horizon == 10:
            temp = 1
        if horizon == 20:
            temp = 9
        elif horizon == 50:
            temp = 121
        pre_model = CNN2(horizon, n_feat, n_classes, temp)
    else:
        print("Model not recognized:", name)
        exit(1)

    # load model
    filename = "pretrained_models/" + name + "_" + str(horizon) + "_0.001_32_best_model.pt"
    pre_model.load_state_dict(torch.load(filename, map_location=DEVICE))
    pre_model.eval()
    pre_model.to(DEVICE)

    return pre_model

def main():
    global WANDB_IN, BATCH_SIZE, lob_data, ths, MIN_VOLUME, stats_fname, WEIGHT_LOSS

    losses_weights = [(.5, .5, 0), (.8, .2, 0), (1, 0, 0), (.3, .3, .3), (.8, .8, 0)]
    with wandb.init(project="adv-gan-fresh") as wandb_instance:
        # {"loss_triplet":0, "window": 20, "horizon": 20, "batch_size": 32}  #wandb_instance.config
        wandb_hyperparams_config = wandb_instance.config

        BATCH_SIZE = wandb_hyperparams_config["batch_size"]
        window_size = wandb_hyperparams_config["window"]
        horizon = wandb_hyperparams_config["horizon"]
        WANDB_IN = wandb_instance

        # pick a regime to try
        lo_regime = losses_weights[wandb_hyperparams_config["loss_triplet"]]
        WEIGHT_LOSS = {"class": lo_regime[0], "gan": lo_regime[1], "cost":  lo_regime[2]}

        #### -------------- #####

        lobster_training = False
        lobster_input_dir = 'indata/MSFT_2020_1'

        # TODO update and use in time (create a script that automatically detect and create them)

        # IF TRUE, we have to use lobster data!!
        add_costpredictor_columns = True
        # very expensive, be aware
        compute_thresholds = True

        # training with lob data
        if lobster_training:
            base_lob_dts = LOBDataset(lobster_input_dir,
                                      horizon=window_size, normalize=True,
                                      add_costpredictor_columns=add_costpredictor_columns,
                                      ratio_rolling_window=-1)
            if compute_thresholds:
               markovian_cost_lobster.compute_npy_cost_matrices(lobster_input_dir)

        # training with deep lob data
        else:
            base_lob_dts = DEEPDataset('indata/FI-2010',
                                       horizon=window_size,
                                       add_costpredictor_columns=add_costpredictor_columns)
            if compute_thresholds:
                markovian_cost_fi.compute_npy_cost_matrices(base_lob_dts)

        # load thresholds according the correct dataset
        th_files = ["prob_exec.npy", "volatilies_quantiles.npy", "volume_quantiles.npy"]
        ths = {fn: None for fn in th_files}
        for fn in ths:
            if lobster_training:
                stock_dir = lobster_input_dir.replace("/", "_")
                npy_dir = "data/thresholds/" + stock_dir + "/"
                with open(npy_dir + fn, 'rb') as f:
                    ths[fn] = np.load(f)
            else:
                with open("data/thresholds_FI/" + fn, 'rb') as f:
                    ths[fn] = np.load(f)

        MIN_VOLUME = base_lob_dts.min_volume()

        lob_data = LOBDataModule(lobdaset=base_lob_dts, window_size=window_size, horizon=horizon)
        rows, columns = lob_data.dims

        stats_final_dict = defaultdict(list)

        attacked_model = "CNN2"
        model = GAN(rows, columns,
                    model=attacked_model,
                    horizon=horizon,
                    latent_dim=100,
                    costpredictor_columns=add_costpredictor_columns,
                    min_volume=MIN_VOLUME,
                    stats_final_dict=stats_final_dict)

        trainer = Trainer(gpus=AVAIL_GPUS,
                          max_epochs=conf.EPOCHS,
                          progress_bar_refresh_rate=20,
                          num_sanity_val_steps=0,
                          check_val_every_n_epoch=conf.VALIDATE_GAN_MODEL_EVERY)

        stats_fname = "exp|bas-{}|hor-{}|win-{}|atk-{}|isfi-{}|wei-{}".format(BATCH_SIZE, horizon, window_size,
                                                                              attacked_model, int(not lobster_training),
                                                                              wandb_hyperparams_config["loss_triplet"])

        trainer.fit(model, lob_data)


def log_stats(stats_fname, stats_dict):
    # saving final stats in a json file
    with open("data/{}.json".format(stats_fname), "w") as out_file:
        json.dump(stats_dict, out_file)


# if __name__ == '__main__':
main()



