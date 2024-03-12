
from src.utils.utils_generic import write_json, is_jsonable
from src.constants import Predictions
import src.constants as cst
from collections import defaultdict


class Metrics:
    def __init__(self, config):
        self._config = config
        self._config_dict = None
        self._testing_metrics = defaultdict(dict)
        self._testing_cf = dict()

    def update_metrics(self, symbol: str, testing_metrics: dict):
        self._testing_metrics[symbol].update(testing_metrics)

    def update_cfm(self, symbol: str, testing_cf):
        self._testing_cf[symbol] = testing_cf

    def dump(self, dir):
        for sym in self._testing_metrics:
            cm = self._testing_cf[sym]
            met = self._testing_metrics[sym]

            # removes keys that are not serializable
            compound_dict = {**met, **self._config_dict, **{"cm": cm.tolist()}}

            keys_to_serialize = [k for k, v in compound_dict.items() if is_jsonable(v)]
            compound_dict = {k: compound_dict[k] for k in keys_to_serialize}

            fname = self._config.cf_name_format('.json').format(
                self._config.PREDICTION_MODEL.name,
                self._config.SEED,
                self._config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name,
                sym,
                self._config.DATASET_NAME.value,
                self._config.CHOSEN_PERIOD.name,
                self._config.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW],
                self._config.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW],
                self._config.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON],
            )
            print("Writing", dir + fname)
            write_json(compound_dict, dir + fname)

    def close(self, dir=cst.DIR_EXPERIMENTS):
        self._config_dict = self._config.__dict__
        self.dump(dir)
