
from src.settings import SettingsExp
import src.constants as cst

# cartesian product of the tests
INDEPENDENT_VARIABLES = {
    SettingsExp.SEED: [0],
    SettingsExp.PREDICTION_MODEL: [cst.Models.CNN1, cst.Models.CNN2],
    SettingsExp.PREDICTION_HORIZON_FUTURE: [10, 5],
    SettingsExp.PREDICTION_HORIZON_PAST: [1],
    SettingsExp.OBSERVATION_PERIOD: [100]
}

# no entry in here = cartesian product of the INDEPENDENT_VARIABLES
# k: v means, when k does not vary, the variable is fixed to v
INDEPENDENT_VARIABLES_CONSTRAINTS = {
    SettingsExp.PREDICTION_MODEL: cst.Models.CNN1,  # when other variables vary, PREDICTION_MODEL = MLP
    SettingsExp.PREDICTION_HORIZON_FUTURE: 5
}
