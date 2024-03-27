
import torch.nn as nn
import torch
from src.models.binctabl.base import BiN, BL_layer, TABL_layer

from src.models.lobcast_model import LOBCAST_model, LOBCAST_module
from src.hyper_parameters import HPTunable


class BinCTABL(LOBCAST_model):
    def __init__(self,
                 input_dim,
                 output_dim,
                 d2, d1, t1, t2, d3, t3, d4, t4):
        super().__init__(input_dim, output_dim)

        self.BiN = BiN(d2, d1, t1, t2)
        self.BL = BL_layer(d2, d1, t1, t2)
        self.BL2 = BL_layer(d3, d2, t2, t3)
        self.TABL = TABL_layer(d4, d3, t3, t4)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # first of all we pass the input to the BiN layer, then we use the C(TABL) architecture
        x = torch.permute(x, (0, 2, 1))

        x = self.BiN(x)

        self.max_norm_(self.BL.W1.data)
        self.max_norm_(self.BL.W2.data)
        x = self.BL(x)
        x = self.dropout(x)

        self.max_norm_(self.BL2.W1.data)
        self.max_norm_(self.BL2.W2.data)
        x = self.BL2(x)
        x = self.dropout(x)

        self.max_norm_(self.TABL.W1.data)
        self.max_norm_(self.TABL.W.data)
        self.max_norm_(self.TABL.W2.data)
        x = self.TABL(x)
        x = torch.squeeze(x)
        x = torch.softmax(x, 1)
        return x

    def max_norm_(self, w):
        with torch.no_grad():
            if (torch.linalg.matrix_norm(w) > 10.0):
                norm = torch.linalg.matrix_norm(w)
                desired = torch.clamp(norm, min=0.0, max=10.0)
                w *= (desired / (1e-8 + norm))


class HP(HPTunable):
    def __init__(self):
        super().__init__()
        self.d1 = {"values": [40]}
        self.d2 = {"values": [60]}
        self.d3 = {"values": [120]}
        self.d4 = {"values": [3]}

        self.t1 = {"values": [10]}
        self.t2 = {"values": [10]}
        self.t3 = {"values": [5]}
        self.t4 = {"values": [1]}


BinCTABL_ml = LOBCAST_module(BinCTABL, HP())
