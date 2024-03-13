# Using Deep Learning to Detect Price Change Indications in Financial Markets
# Source: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8081663

import pytorch_lightning as pl
from torch import nn
import src.utils.utils_generic as utils_generic


# subset of the arguments of a LOBCAST_model
CONFIG = {
    "hidden_layer_dim": [128],
    "p_dropout": [.4, .1],
}


class LOBCAST_module:
    def __init__(self, model, tunable_parameters: dict):
        self.model = model
        self.tunable_parameters = tunable_parameters

        # self.__check_parameters()

    # def __check_parameters(self):
    #     list_arguments = utils_generic.get_class_arguments(self.model)
        # arguments = set(list_arguments)
        # tunable = set(self.tunable_parameters.keys())

        # is_sub = tunable.issubset(arguments)
        # excess = tunable - arguments
        # if not is_sub:
        #     raise ValueError(f"\nThe following hps you are trying to tune, are not arguments of your model."
        #                      f" Make sure that arguments in tunable_parameters are a subset of those in your model.\n"
        #                      f" > Arguments in your model: {arguments}\n"
        #                      f" > Argument to optimize: {excess}")
        # else:
        #     print(f"OK. Tuning hps {self.tunable_parameters}")


class LOBCAST_model(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim


class MLP(LOBCAST_model):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_layer_dim,
            p_dropout
    ):
        super().__init__(input_dim, output_dim)

        self.linear1 = nn.Linear(self.input_dim, hidden_layer_dim)
        self.leakyReLU = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=p_dropout)
        self.linear2 = nn.Linear(hidden_layer_dim, self.output_dim)

    def forward(self, x):
        # [batch_size x 40 x observation_length]
        x = x.view(x.size(0), -1).float()
        out = self.linear1(x)
        out = self.leakyReLU(out)
        out = self.dropout(out)
        out = self.linear2(out)
        return out


MLP_lm = LOBCAST_module(MLP, CONFIG)
