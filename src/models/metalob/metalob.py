import pytorch_lightning as pl
from torch import nn
import src.constants as cst
import torch


class MetaLOB(pl.LightningModule):
    def __init__(self, mlp_hidden, chosen_models):
        super().__init__()
        self.input_dim = len(chosen_models) * cst.NUM_CLASSES

        self.fc1 = nn.Linear(self.input_dim, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, cst.NUM_CLASSES)

        self.batch_norm = nn.BatchNorm1d(mlp_hidden)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = x.reshape(x.shape[0], self.input_dim)
        # x.shape = [batch, n_models*n_classes]

        o = self.fc1(x)
        # o.shape = [batch, mlp_hidden]

        o = self.batch_norm(self.relu(o))
        # o.shape = [batch, mlp_hidden]

        o = self.fc2(o)
        # o.shape = [batch, n_classes]

        return o

class MetaLOB2(pl.LightningModule):
    def __init__(self, meta_hidden):
        super().__init__()
        input_dim = (len(cst.Models)-1) * cst.NUM_CLASSES
        self.n_models = len(cst.Models)-1

        weight = torch.Tensor(cst.NUM_CLASSES, self.n_models)
        self.W = nn.Parameter(weight)
        nn.init.xavier_uniform(self.W)

        #self.relu = nn.ReLU()

    def forward(self, x):

        output = torch.einsum('bij,ij->bi', (x, self.W))

        return output