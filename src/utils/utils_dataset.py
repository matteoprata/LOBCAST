
import numpy as np

import src.constants as cst
from src.config import Configuration
from src.data_preprocessing.META.METADataset import MetaDataset
from src.data_preprocessing.META.METADataBuilder import MetaDataBuilder


# DATASETS
from src.data_preprocessing.FI.FIDataBuilder import FIDataBuilder
from src.data_preprocessing.FI.FIDataset import FIDataset
from src.data_preprocessing.DataModule import DataModule
from src.data_preprocessing.LOB.LOBDataset import LOBDataset


def prepare_data_fi(config: Configuration):

    fi_train = FIDataBuilder(
        cst.DATA_SOURCE + cst.DATASET_FI,
        dataset_type=cst.DatasetType.TRAIN,
        horizon=config.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON],
        window=config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_SNAPSHOTS],
        train_val_split=config.TRAIN_SPLIT_VAL,
        chosen_model=config.CHOSEN_MODEL
    )

    fi_val = FIDataBuilder(
        cst.DATA_SOURCE + cst.DATASET_FI,
        dataset_type=cst.DatasetType.VALIDATION,
        horizon=config.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON],
        window=config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_SNAPSHOTS],
        train_val_split=config.TRAIN_SPLIT_VAL,
        chosen_model=config.CHOSEN_MODEL
    )

    fi_test = FIDataBuilder(
        cst.DATA_SOURCE + cst.DATASET_FI,
        dataset_type=cst.DatasetType.TEST,
        horizon=config.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON],
        window=config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_SNAPSHOTS],
        chosen_model=config.CHOSEN_MODEL
    )

    train_set = FIDataset(
        x=fi_train.get_samples_x(),
        y=fi_train.get_samples_y(),
        chosen_model=config.CHOSEN_MODEL,
        num_snapshots=config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_SNAPSHOTS],
    )
    perc_cl = lambda a: np.array(list(a.values())) / sum(a.values())

    # print() HAS PROBLEMS WITH DEEPLOBATT
    # print("TRAIN balance", Counter(fi_train.get_samples_y()), perc_cl(Counter(fi_train.get_samples_y())))

    val_set = FIDataset(
        x=fi_val.get_samples_x(),
        y=fi_val.get_samples_y(),
        chosen_model=config.CHOSEN_MODEL,
        num_snapshots=config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_SNAPSHOTS],
    )
    # print("VAL balance", Counter(fi_val.get_samples_y()), perc_cl(Counter(fi_val.get_samples_y())))

    test_set = FIDataset(
        x=fi_test.get_samples_x(),
        y=fi_test.get_samples_y(),
        chosen_model=config.CHOSEN_MODEL,
        num_snapshots=config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_SNAPSHOTS],
    )

    # print("TEST balance", Counter(fi_test.get_samples_y()), perc_cl(Counter(fi_test.get_samples_y())))
    # print()

    fi_dm = DataModule(
        train_set, val_set, test_set,
        config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE],
        config.HYPER_PARAMETERS[cst.LearningHyperParameter.IS_SHUFFLE_TRAIN_SET]
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

    if config.CHOSEN_MODEL != cst.Models.DEEPLOBATT:

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


def prepare_data_meta(config: Configuration):

    if config.CHOSEN_PERIOD == cst.Periods.FI:
        databuilder_test = FIDataBuilder(
            cst.DATA_SOURCE + cst.DATASET_FI,
            dataset_type=cst.DatasetType.TEST,
            horizon=config.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON],
            window=config.HYPER_PARAMETERS[cst.LearningHyperParameter.NUM_SNAPSHOTS],
            chosen_model=config.CHOSEN_MODEL
        )
        truth_y = databuilder_test.samples_y[100:]
    else:
        train_set = LOBDataset(
            config=config,
            dataset_type=cst.DatasetType.TRAIN,
            stocks_list=config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].value,
            start_end_trading_day=config.CHOSEN_PERIOD.value['train']
        )

        vol_price_mu, vol_price_sig = train_set.vol_price_mu, train_set.vol_price_sig

        test_set = LOBDataset(
            config=config,
            dataset_type=cst.DatasetType.TEST,
            stocks_list=config.CHOSEN_STOCKS[cst.STK_OPEN.TEST].value,
            start_end_trading_day=config.CHOSEN_PERIOD.value['test'],
            vol_price_mu=vol_price_mu, vol_price_sig=vol_price_sig
        )
        truth_y = test_set.y.numpy()
        truth_y = truth_y[100:]

    meta_databuilder = MetaDataBuilder(
        truth_y=truth_y,
        config=config
    )

    train_set = MetaDataset(
        meta_databuilder.get_samples_train(),
        dataset_type=cst.DatasetType.TRAIN,
        num_classes=cst.NUM_CLASSES
    )

    val_set = MetaDataset(
        meta_databuilder.get_samples_val(),
        dataset_type=cst.DatasetType.VALIDATION,
        num_classes=cst.NUM_CLASSES
    )

    test_set = MetaDataset(
        meta_databuilder.get_samples_test(),
        dataset_type=cst.DatasetType.TEST,
        num_classes=cst.NUM_CLASSES
    )

    print("Samples in the splits:")
    print(len(train_set), len(val_set), len(test_set))
    print()

    meta_dm = DataModule(
        train_set, val_set, test_set,
        config.HYPER_PARAMETERS[cst.LearningHyperParameter.BATCH_SIZE],
        config.HYPER_PARAMETERS[cst.LearningHyperParameter.IS_SHUFFLE_TRAIN_SET]
    )
    return meta_dm


def pick_dataset(config: Configuration):

    if config.CHOSEN_DATASET == cst.DatasetFamily.LOB:
        return prepare_data_lob(config)

    elif config.CHOSEN_DATASET == cst.DatasetFamily.FI:
        return prepare_data_fi(config)

    elif config.CHOSEN_DATASET == cst.DatasetFamily.META:
        return prepare_data_meta(config)
