
from src.utils.utilities import write_json, is_jsonable
from src.constants import Predictions
import src.constants as cst


class Metrics:
    def __init__(self, config):
        self._config = config
        self._config_dict = None
        self._testing_metrics = []  # list of tuples (test symbol, metric)
        self._testing_cf = []       # list of tuples (test symbol, cfm)

    def add_testing_metrics(self, symbol, testing_metrics):
        self._testing_metrics += [(symbol, testing_metrics)]

    def add_testing_cfm(self, symbol, testing_cf):
        self._testing_cf += [(symbol, testing_cf)]

    def dump(self):
        for isym in range(len(self._testing_metrics)):
            sym, cm = self._testing_cf[isym]
            _, met = self._testing_metrics[isym]

            # removes keys that are not serializable
            compound_dict = {**met, **{"cm": cm.tolist()}, **self._config_dict}
            keys_to_serialize = [k for k, v in compound_dict.items() if is_jsonable(v)]
            compound_dict = {k: compound_dict[k] for k in keys_to_serialize}

            fname = self._config.cf_name_format('.json').format(
                self._config.CHOSEN_MODEL.name,
                self._config.SEED,
                self._config.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name,
                sym,
                self._config.CHOSEN_DATASET.value,
                self._config.CHOSEN_PERIOD.name,
                self._config.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW],
                self._config.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW],
                self._config.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON],
            )
            write_json(compound_dict, cst.DIR_EXPERIMENTS + fname)
            print("DUMPING", cst.DIR_EXPERIMENTS + fname)

    def close(self):
        self._config_dict = self._config.__dict__
        self.dump()
