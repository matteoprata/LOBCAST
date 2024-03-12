# Time-series classification using neural bag-of-features
# Source: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8081217&casa_token=XxAI2VwaYlcAAAAA:caTIvkUR3wIrhqg00IJVogdyhG_lq988lY7e9Frbqv2SmCgoHZrrpp-bLB1ZpOmyCZ0gdUwLGw&tag=1

import src.constants as cst
import pytorch_lightning as pl
from torch import nn
import torch


class NBoF(pl.LightningModule):
    def __init__(self, num_snapshots, num_features, num_rbf_neurons, hidden_mlp, centers=None, lr_W=None):
        super().__init__()
        self.num_snapshots = num_snapshots
        self.num_rbf_neurons = num_rbf_neurons
        self.lr_W = lr_W

        # initialize with centers
        centers = centers if centers is not None else torch.Tensor(num_rbf_neurons, num_features)
        self.V = nn.Parameter(centers)

        g = 0.5  # (0.1, 10)
        # wt = (g[0] - g[1]) * torch.rand(num_rbf_neurons, num_features) + g[1]
        wt = torch.ones(num_rbf_neurons, num_features) * g
        self.W = nn.Parameter(wt)

        self.softmax = nn.Softmax(dim=0)

        # self.W_h = nn.Linear(num_rbf_neurons, hidden_mlp)  # MLP
        # self.W_o = nn.Linear(hidden_mlp, 3)                # MLP
        #
        # self.elu = nn.ELU()

        self.base = nn.Sequential(
            nn.Linear(num_rbf_neurons, hidden_mlp),
            nn.ELU(),
            nn.Linear(hidden_mlp, 3)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        # x.shape = [batch_size, observation_length, num_features]

        inputs = torch.zeros(batch_size, self.num_rbf_neurons, self.num_snapshots, device=cst.DEVICE_TYPE)
        for k, (v, w) in enumerate(zip(self.V, self.W)):
            # v.shape = [40]
            v = v.unsqueeze(0).unsqueeze(0)
            v = v.repeat(batch_size, self.num_snapshots, 1)
            # v.shape = [batch_size, observation_length, num_features]

            i = (x - v) * w
            # i.shape = [batch_size, observation_length, num_features]

            inputs[:, k, :] = - torch.linalg.norm(i, dim=2, ord=2)
            # inputs[k].shape = [batch_size, observation_length]

        # inputs.shape = [batch_size, num_rbf_neurons, observation_length]

        phi = self.softmax(inputs)
        # phi.shape = [batch_size, num_rbf_neurons, observation_length]

        s = torch.mean(phi, dim=2)
        # s.shape = [batch_size, num_rbf_neurons]

        out = self.base(s)
        # phi.shape = [batch_size, hidden_mlp]
        # phi.shape = [batch_size, hidden_mlp]
        # phi.shape = [batch_size, 3]

        return out
