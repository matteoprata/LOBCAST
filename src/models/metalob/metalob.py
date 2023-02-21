import pytorch_lightning as pl
from torch import nn
import src.constants as cst


class MetaLOB(pl.LightningModule):
    def __init__(self, mlp_hidden):
        super().__init__()
        input_dim = len(cst.Models) * cst.NUM_CLASSES

        self.fc1 = nn.Linear(input_dim, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, cst.NUM_CLASSES)

        self.batch_norm = nn.BatchNorm1d(mlp_hidden)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x.shape = [batch, n_models*n_classes]

        o = self.fc1(x)
        # o.shape = [batch, mlp_hidden]

        o = self.batch_norm(self.relu(o))
        # o.shape = [batch, mlp_hidden]

        o = self.fc2(o)
        # o.shape = [batch, n_classes]

        return o
