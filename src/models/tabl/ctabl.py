# Temporal Attention augmented Bilinear Network for Financial Time-Series Data Analysis
# Source: https://ieeexplore.ieee.org/abstract/document/8476227

import pytorch_lightning as pl
from torch import nn
import torch
from src.models.tabl.bl_layer import BL_layer
from src.models.tabl.tabl_layer import TABL_layer


class CTABL(nn.Module):
    def __init__(self, d2, d1, t1, t2, d3, t3, d4, t4):
        super().__init__()

        self.BL = BL_layer(d2, d1, t1, t2)
        self.BL2 = BL_layer(d3, d2, t2, t3)
        self.TABL = TABL_layer(d4, d3, t3, t4)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
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

        return x

    def max_norm_(self, w):
        with torch.no_grad():
            if (torch.linalg.matrix_norm(w) > 10.0):
                norm = torch.linalg.matrix_norm(w)
                desired = torch.clamp(norm, min=0.0, max=10.0)
                w *= (desired / (1e-8 + norm))
