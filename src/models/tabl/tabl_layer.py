import src.config as co
import pytorch_lightning as pl
import torch
from torch import nn

class TABL_layer(pl.LightningModule):

    def __init__(self, d2, d1, t1, t2):
        super().__init__()
        self.t1 = t1

        weight = torch.Tensor(d2, d1)
        self.W1 = nn.Parameter(weight)
        nn.init.kaiming_uniform_(self.W1, nonlinearity='relu')

        weight2 = torch.Tensor(t1, t1)
        self.W2 = nn.Parameter(weight2)
        nn.init.constant_(self.W2, 1 / t1)

        weight3 = torch.Tensor(t1, t2)
        self.W3 = nn.Parameter(weight3)
        nn.init.kaiming_uniform_(self.W3, nonlinearity='relu')

        bias1 = torch.Tensor(d2, t2)
        self.B = nn.Parameter(bias1)
        nn.init.constant_(self.B, 0)

        l = torch.Tensor(1, )
        self.l = nn.Parameter(l)
        nn.init.constant_(self.l, 0.5)

        self.activation = nn.ReLU()

    def forward(self, X):

        # print(self.l)
        if self.l[0] < 0:
            l = torch.Tensor(1, )
            self.l = nn.Parameter(l)
            nn.init.constant_(self.l, 0.0)

        if self.l[0] > 1:
            l = torch.Tensor(1, )
            self.l = nn.Parameter(l)
            nn.init.constant_(self.l, 1.0)

        # print(self.l[0])
        X = self.W1 @ X

        # enforcing constant (1) on the diagonal
        W2 = \
            self.W2 \
            - self.W2 \
            * torch.eye(
                self.t1,
                dtype=torch.float32,
                device=co.DEVICE_TYPE
            )\
            + torch.eye(
                self.t1,
                dtype=torch.float32,
                device=co.DEVICE_TYPE)\
            / self.t1

        E = X @ self.W2

        # print(E.shape)
        A = torch.softmax(E, dim=-1)
        X = self.l[0] * (X) + (1.0 - self.l[0]) * X * A
        y = X @ self.W3 + self.B

        return y