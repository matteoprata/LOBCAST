# Using Deep Learning to Detect Price Change Indications in Financial Markets
# Source: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8081663

from torch import nn
from src.models.lobcast_model import LOBCAST_model, LOBCAST_module
import src.utils.utils_generic as utils_generic


# subset of the arguments of a LOBCAST_model
CONFIG = {
    "hidden_layer_dim": [128],
    "p_dropout": [.4, .1],
}


class MLP(LOBCAST_model):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_layer_dim,
            p_dropout
    ):
        super().__init__(input_dim, output_dim)

        flat_dims = self.input_dim[0] * self.input_dim[1]
        self.linear1 = nn.Linear(flat_dims, hidden_layer_dim)
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


MLP_lm = LOBCAST_module("MLP", MLP, CONFIG)
