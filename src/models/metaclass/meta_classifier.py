import pytorch_lightning as pl
from torch import nn
import torch
import src.constants as cst


# meta classifier class
class MetaLOB(pl.LightningModule):
    def __init__(self, n_classifiers, num_classes, dim_midlayer):
        super().__init__()
        input_dim = n_classifiers * num_classes
        self.fc1 = nn.Linear(input_dim, dim_midlayer)
        self.batch = nn.BatchNorm1d(dim_midlayer)
        self.fc2 = nn.Linear(dim_midlayer, num_classes)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.batch(self.activation(self.fc1(x)))
        x = self.fc2(x)
        return x
