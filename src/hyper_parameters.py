
from src.utils.utils_generic import dict_to_string


class ConfigHP:
    def add_hyperparameters(self, params: dict):
        for key, value in params.items():
            self.__setattr__(key, value)

    def add_hyperparameter(self, key, value):
        self.__setattr__(key, value)

    def __repr__(self):
        return dict_to_string(self.__dict__)


class ConfigHPTuned(ConfigHP):
    pass


class ConfigHPTunable(ConfigHP):
    def __init__(self):
        self.BATCH_SIZE = {"values": [32, 64]}         # {"min": 0.0001, "max": 0.1} or {"values": [11]}
        self.LEARNING_RATE = {"values": [0.0001, 0.001, 0.01]}  # {"min": 0.0001, "max": 0.1}  # {"min": 0.0001, "max": 0.1}
        self.OPTIMIZER = {"values": ["SGD"]}


