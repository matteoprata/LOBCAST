
import argparse
from src.utils.lobdataset import LOBDataset
import numpy as np
import torch
from collections import defaultdict
from collections import Counter
from src.models.cnn2_evolution import CNN2
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from src.data_preprocessing.LOBDataBuilder import LOBDataBuilder
from src.data_preprocessing.LOBDataset import LOBDataset

import src.config as co


def parser_cl_arguments():
    """ parses the arguments for the command line. """
    parser = argparse.ArgumentParser(description='PyTorch Training')

    # parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    # parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N', help='mini-batch size')
    # parser.add_argument('-k', '--horizon', default=100, type=int, metavar='N', help='horizon can be 10, 20, 50 or 100', choices=[10, 20, 50, 100])
    # parser.add_argument('-m', '--type_model', default="LSTM", type=str, help='Name of the type_model, can be MLP, CNN, CNN2, LSTM, DeepLob or CNN_LSTM', choices=["MLP", "CNN", "CNN2", "DeepLob", "LSTM", "CNN_LSTM"]
    # parser.add_argument('-l', '--learning_rate', default=1e-3, type=float, help='The learning rate of the saved_models')
    return parser


def load_data(batch_size, base_lob_dts, type_model) -> tuple:
    """ Return dataset_train, dataset_val, dataset_test """

    is_torch_dataset = True
    dataset_train = base_lob_dts.split_train_data(torch_dataset=is_torch_dataset)
    dataset_val = base_lob_dts.split_val_data(torch_dataset=is_torch_dataset)
    dataset_test = base_lob_dts.split_test_data(torch_dataset=is_torch_dataset)
    del base_lob_dts

    print("training classes distribution", dataset_train.counts)
    print(type(dataset_train.x))

    # ------ DO UNDER-SAMPLING (without replacement)

    minority_quantities = min(dataset_train.counts)
    minority_class = [k for k, v in enumerate(dataset_train.counts) if dataset_train.counts[k] == minority_quantities][0]
    print("minority_quantities", minority_quantities, "minority_class", minority_class)

    # map CLASS: OBSERVATIONS
    per_class_dict = defaultdict(list)
    for i, (x, y) in enumerate(dataset_train):
        per_class_dict[y.item()].append(x)

    # minority class takes it all Q
    trainX, trainY = [], []
    trainX += [t.numpy() for t in per_class_dict[minority_class]]
    trainY += [minority_class for _ in range(len(per_class_dict[minority_class]))]

    # all other classes take Q random each
    for k in per_class_dict:
        if k != minority_class:
            idxs = np.random.choice(len(per_class_dict[k]), minority_quantities)
            trainX += [per_class_dict[k][i].numpy() for i in idxs]
            trainY += [k for _ in range(len(per_class_dict[minority_class]))]

    x = np.asarray(trainX)
    y = np.asarray(trainY)

    dataset_train.reset_x_y(x, y)
    print("classes distribution:", dataset_train.counts)

    # ------ DONE UNDER-SAMPLING (without replacement)

    # create dataloaders
    # sampler option is mutually exclusive with shuffle
    train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, drop_last=True, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader  = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader


def run_training(horizon, train_loader, val_loader, test_loader):
    model = CNN2(horizon, 40, 3, 249)

    AVAIL_GPUS = min(1, torch.cuda.device_count())
    EPOCHS_VALIDATE = 10
    EPOCHS = 100

    trainer = Trainer(gpus=AVAIL_GPUS,
                      max_epochs=EPOCHS,
                      progress_bar_refresh_rate=20,
                      num_sanity_val_steps=0,
                      check_val_every_n_epoch=EPOCHS_VALIDATE)

    trainer.fit(model, train_loader)
    # trainer.fit(model, val_loader)
    # trainer.fit(model, test_loader)


# if __name__ == "__main__":
#     args = parser_cl_arguments().parse_args()
#
#     # dataset args
#     # dir_data = "data/AVXL_010322_310322"
#
#     # type_model = args.type_model
#     # n_epochs = args.epochs
#     # batch_size = args.batch_size
#     # learning_rate = args.learning_rate
#
#     # 0. LOAD data
#     print("here\n\n")
#
#     # train
#     lo_train = LOBDataBuilder(co.DATA_DIR,
#                               co.DatasetType.TRAIN,
#                               start_end_trading_day=("2022-03-07", "2022-03-08"),
#                               is_data_preload=False,
#                               crop_trading_day_by=60*30)
#
#     mu, sigma = lo_train.normalization_mean, lo_train.normalization_std
#
#     lo_test = LOBDataBuilder(co.DATA_DIR,
#                              co.DatasetType.TEST,
#                              start_end_trading_day=("2022-03-09", "2022-03-10"),
#                              is_data_preload=False,
#                              crop_trading_day_by=60*30,
#                              normalization_mean=mu,
#                              normalization_std=sigma)
#
#     n_inst_train = int(len(lo_train.samples_x) * .7)
#
#     train_set = LOBDataset(x=lo_train.samples_x[:n_inst_train], y=lo_train.samples_y[:n_inst_train])
#     val_set   = LOBDataset(x=lo_train.samples_x[n_inst_train:], y=lo_train.samples_y[n_inst_train:])
#     test_set  = LOBDataset(x=lo_test.samples_x, y=lo_test.samples_y)
#
#     # generates the dataset normalized and balanced
#     base_lob_dts = LOBDataset(co.DATA_DIR, horizon=co.HISTORIC_WIN_SIZE, sign_threshold=0.1)
#
#     print("ready, ", type(base_lob_dts))
#
#     # 1. SPLIT DATA
#     train_loader, val_loader, test_loader = load_data(batch_size, base_lob_dts, type_model)
#
#     print("OK")
#     # run_training(horizon, train_loader, val_loader, test_loader)
