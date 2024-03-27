
from src.utils.utils_generic import dict_to_string


class Hyperparameters:
    def add_hyperparameters(self, params: dict):
        for key, value in params.items():
            self.__setattr__(key, value)

    def add_hyperparameter(self, key, value):
        self.__setattr__(key, value)

    def __repr__(self):
        return dict_to_string(self.__dict__)


class HPTuned(Hyperparameters):
    """ Tuned hyperparameters of the models. Hyperparameters are assigned with their chosen value
    by an external scheduler (e.g. wandb grid search)."""

    def update_hyperparameter(self, hp, value):
        try:
            self.__getattribute__(hp)
            self.__setattr__(hp, value)

        except AttributeError:
            raise AttributeError(f"This class has no {hp} to set.")


class HPTunable(Hyperparameters):
    """ Tunable hyperparameters of the models. Contains the domains of hyperparameters exploration. """
    def __init__(self):
        self.BATCH_SIZE = {"values": [32, 64]}   # {"min": 0.0001, "max": 0.1} or {"values": [11]}
        self.LEARNING_RATE = {"values": [0.0001, 0.001, 0.01]}  # {"min": 0.0001, "max": 0.1}  # {"min": 0.0001, "max": 0.1}
        self.OPTIMIZER = {"values": ["SGD"]}
