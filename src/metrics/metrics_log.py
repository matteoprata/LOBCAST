
from src.utils.utils_generic import write_json, is_jsonable
from src.constants import Predictions
import src.constants as cst
from collections import defaultdict


class Metrics:
    def __init__(self, config, h_parameters):
        self.config = config
        self.h_parameters = h_parameters
        self.metrics = defaultdict(dict)  # dict logged every X epochs

    def add_metric(self, epoch, dataset_type, eval_dict):
        self.metrics[dataset_type][epoch] = eval_dict

    def dump(self, path):
        print("Writing", path)
        merged = {**self.config, **self.h_parameters}
        merged = {k: (v if is_jsonable(v) else str(v)) for k, v in merged.items()}  # make string unserializable vals

        write_json(merged, path + "config.json")
        write_json(self.metrics, path + "metrics.json")
