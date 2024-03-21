
from src.settings import SettingsExp
import src.constants as cst


INDEPENDENT_VARIABLES = {
    SettingsExp.SEED: [0],
    SettingsExp.PREDICTION_MODEL: [cst.ModelsClass.MLP, cst.ModelsClass.CNN1, cst.ModelsClass.CNN2],
    SettingsExp.PREDICTION_HORIZON_FUTURE: [5],
    SettingsExp.PREDICTION_HORIZON_PAST: [1],
    SettingsExp.OBSERVATION_PERIOD: [100]
}

INDEPENDENT_VARIABLES_CONSTRAINTS = {
    SettingsExp.PREDICTION_MODEL: cst.ModelsClass.MLP,
    SettingsExp.PREDICTION_HORIZON_FUTURE: 5
}
