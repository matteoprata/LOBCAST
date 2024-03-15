
import numpy as np

import src.constants as cst
from src.config import Configuration
from src.data_preprocessing.META.METADataset import MetaDataset
from src.data_preprocessing.META.METADataBuilder import MetaDataBuilder


# DATASETS
from src.data_preprocessing.FI.FIDataBuilder import FIDataset
# from src.data_preprocessing.FI.FIDataset import FIDataset
from src.data_preprocessing.DataModule import DataModule
from src.data_preprocessing.LOB.LOBDataset import LOBDataset


def prepare_data_fi(config: Configuration):

    fi_train = FIDataset(
        cst.DATA_SOURCE + cst.DATASET_FI,
        dataset_type=cst.DatasetType.TRAIN,
        horizon=config.PREDICTION_HORIZON_FUTURE,
        observation_length=config.OBSERVATION_PERIOD,
        train_val_split=config.TRAIN_SET_PORTION,
        n_trends=config.N_TRENDS
    )

    fi_val = FIDataset(
        cst.DATA_SOURCE + cst.DATASET_FI,
        dataset_type=cst.DatasetType.VALIDATION,
        horizon=config.PREDICTION_HORIZON_FUTURE,
        observation_length=config.OBSERVATION_PERIOD,
        train_val_split=config.TRAIN_SET_PORTION,
        n_trends=config.N_TRENDS
    )

    fi_test = FIDataset(
        cst.DATA_SOURCE + cst.DATASET_FI,
        dataset_type=cst.DatasetType.TEST,
        observation_length=config.OBSERVATION_PERIOD,
        horizon=config.PREDICTION_HORIZON_FUTURE,
        train_val_split=config.TRAIN_SET_PORTION,
        n_trends=config.N_TRENDS
    )

    fi_dm = DataModule(
        fi_train, fi_val, fi_test,
        config.TUNED_H_PRAM.BATCH_SIZE,
        config.IS_SHUFFLE_TRAIN_SET
    )
    return fi_dm


def prepare_data_lob(config: Configuration):

    train_set = LOBDataset(
        config=config,
        dataset_type=cst.DatasetType.TRAIN,
        stocks_list=config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].value,
        start_end_trading_day=config.CHOSEN_PERIOD.value['train']
    )

    vol_price_mu, vol_price_sig = train_set.vol_price_mu, train_set.vol_price_sig

    val_set = LOBDataset(
        config=config,
        dataset_type=cst.DatasetType.VALIDATION,
        stocks_list=config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].value,
        start_end_trading_day=config.CHOSEN_PERIOD.value['val'],
        vol_price_mu=vol_price_mu, vol_price_sig=vol_price_sig
    )

    test_set = LOBDataset(
        config=config,
        dataset_type=cst.DatasetType.TEST,
        stocks_list=config.CHOSEN_STOCKS[cst.STK_OPEN.TEST].value,
        start_end_trading_day=config.CHOSEN_PERIOD.value['test'],
        vol_price_mu=vol_price_mu, vol_price_sig=vol_price_sig
    )

    if config.PREDICTION_MODEL != cst.Models.DEEPLOBATT:

        print()
        print()
        print()

        train_occ = np.asarray([train_set.ys_occurrences[0.0], train_set.ys_occurrences[1.0], train_set.ys_occurrences[2.0]])
        val_occ = np.asarray([val_set.ys_occurrences[0.0], val_set.ys_occurrences[1.0], val_set.ys_occurrences[2.0]])
        test_occ = np.asarray([test_set.ys_occurrences[0.0], test_set.ys_occurrences[1.0], test_set.ys_occurrences[2.0]])

        train_occ = np.round(train_occ / np.sum(train_occ), 2)
        val_occ = np.round(val_occ / np.sum(val_occ), 2)
        test_occ = np.round(test_occ / np.sum(test_occ), 2)

        print(
            f'Backward: {config.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW]}\t',
            f'Forward: {config.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW]}\t',
            f'Alfa: {cst.ALPHA}'
        )
        print("train:\t", train_occ[0], '\t', train_occ[1], '\t', train_occ[2])
        print("val:\t",   val_occ[0], '\t', val_occ[1], '\t', val_occ[2])
        print("test:\t",  test_occ[0], '\t', test_occ[1], '\t', test_occ[2])

    print("Samples in the splits:")
    print(len(train_set), len(val_set), len(test_set))
    print()

    lob_dm = DataModule(
        train_set, val_set, test_set,
        config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE],
        config.HYPER_PARAMETERS[cst.LearningHyperParameter.IS_SHUFFLE_TRAIN_SET]
    )

    return lob_dm


def pick_dataset(config):

    if config.DATASET_NAME == cst.DatasetFamily.LOB:
        return prepare_data_lob(config)

    elif config.DATASET_NAME == cst.DatasetFamily.FI:
        return prepare_data_fi(config)
