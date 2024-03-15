import src.constants as cst
from src.config import LOBCASTSetupRun
from src.data_preprocessing.DataModule import DataModule
from src.data_preprocessing.FI.FIDataBuilder import FIDataset


def prepare_data_fi(config: LOBCASTSetupRun):
    fi_train, fi_val, fi_test = None, None, None

    if not config.SETTINGS.IS_TEST_ONLY:
        fi_train = FIDataset(
            cst.DATA_SOURCE + cst.DATASET_FI,
            dataset_type=cst.DatasetType.TRAIN,
            horizon=config.SETTINGS.PREDICTION_HORIZON_FUTURE,
            observation_length=config.SETTINGS.OBSERVATION_PERIOD,
            train_val_split=config.SETTINGS.TRAIN_SET_PORTION,
            n_trends=config.SETTINGS.N_TRENDS
        )

        fi_val = FIDataset(
            cst.DATA_SOURCE + cst.DATASET_FI,
            dataset_type=cst.DatasetType.VALIDATION,
            horizon=config.SETTINGS.PREDICTION_HORIZON_FUTURE,
            observation_length=config.SETTINGS.OBSERVATION_PERIOD,
            train_val_split=config.SETTINGS.TRAIN_SET_PORTION,
            n_trends=config.SETTINGS.N_TRENDS
        )

    fi_test = FIDataset(
        cst.DATA_SOURCE + cst.DATASET_FI,
        dataset_type=cst.DatasetType.TEST,
        observation_length=config.SETTINGS.OBSERVATION_PERIOD,
        horizon=config.SETTINGS.PREDICTION_HORIZON_FUTURE,
        train_val_split=config.SETTINGS.TRAIN_SET_PORTION,
        n_trends=config.SETTINGS.N_TRENDS
    )

    fi_dm = DataModule(
        fi_train, fi_val, fi_test,
        config.TUNED_H_PRAM.BATCH_SIZE,
        config.SETTINGS.DEVICE,
        config.SETTINGS.IS_SHUFFLE_TRAIN_SET
    )
    return fi_dm


def pick_dataset(config):
    if config.SETTINGS.DATASET_NAME == cst.DatasetFamily.FI:
        return prepare_data_fi(config)
