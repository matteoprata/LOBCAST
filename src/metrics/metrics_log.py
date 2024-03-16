
from src.utils.utils_generic import write_json, is_jsonable
from src.constants import Predictions
import src.constants as cst
from collections import defaultdict


class Metrics:
    def __init__(self, config, h_parameters, path):
        self.config = config
        self.h_parameters = h_parameters
        self.metrics = defaultdict(dict)  # dict logged every X epochs
        self.path = path

    def add_metric(self, epoch, dataset_type, eval_dict):
        self.metrics[dataset_type][epoch] = eval_dict

    def reset_stats(self):
        self.metrics = defaultdict(dict)

    def dump_info(self):
        print("Dumping config at", self.path)
        merged = {**self.config, **self.h_parameters}
        merged = {k: (v if is_jsonable(v) else str(v)) for k, v in merged.items()}  # make string unserializable vals
        write_json(merged, self.path + "config.json")

    def dump_metrics(self, fname):
        print("Dumping metrics at", self.path)
        write_json(self.metrics, self.path + fname)
