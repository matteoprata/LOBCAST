import src.constants as cst
from src.data_preprocessing.dataModule import DataModule
from src.data_preprocessing.FI.FIDataBuilder import FIDataset


def prepare_data_fi(sim):
    fi_train, fi_val, fi_test = None, None, None

    if not sim.SETTINGS.IS_TEST_ONLY:
        fi_train = FIDataset(
            cst.DATASET_FI,
            dataset_type=cst.DatasetType.TRAIN,
            horizon=sim.SETTINGS.PREDICTION_HORIZON_FUTURE,
            observation_length=sim.SETTINGS.OBSERVATION_PERIOD,
            train_val_split=sim.SETTINGS.TRAIN_SET_PORTION,
            n_trends=sim.SETTINGS.N_TRENDS
        )

        fi_val = FIDataset(
            cst.DATASET_FI,
            dataset_type=cst.DatasetType.VALIDATION,
            horizon=sim.SETTINGS.PREDICTION_HORIZON_FUTURE,
            observation_length=sim.SETTINGS.OBSERVATION_PERIOD,
            train_val_split=sim.SETTINGS.TRAIN_SET_PORTION,
            n_trends=sim.SETTINGS.N_TRENDS
        )

    fi_test = FIDataset(
        cst.DATASET_FI,
        dataset_type=cst.DatasetType.TEST,
        observation_length=sim.SETTINGS.OBSERVATION_PERIOD,
        horizon=sim.SETTINGS.PREDICTION_HORIZON_FUTURE,
        train_val_split=sim.SETTINGS.TRAIN_SET_PORTION,
        n_trends=sim.SETTINGS.N_TRENDS
    )

    fi_dm = DataModule(
        fi_train, fi_val, fi_test,
        sim.HP_TUNED.BATCH_SIZE,
        sim.SETTINGS.DEVICE,
        sim.SETTINGS.IS_SHUFFLE_TRAIN_SET
    )
    return fi_dm


def pick_dataset(sim):
    if sim.SETTINGS.DATASET_NAME == cst.DatasetFamily.FI:
        return prepare_data_fi(sim)
    else:
        raise ValueError(f"Unhandled dataset name: {sim.SETTINGS}")
