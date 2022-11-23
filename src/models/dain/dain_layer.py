import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class DAIN_Layer(nn.Module):
    def __init__(
        self,
        mode,
        mean_lr,
        gate_lr,
        scale_lr,
        input_dim
    ):
        super(DAIN_Layer, self).__init__()
        # print("Mode = ", mode)

        self.mode = mode
        self.mean_lr = mean_lr
        self.gate_lr = gate_lr
        self.scale_lr = scale_lr

        # Parameters for adaptive average
        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.mean_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive std
        self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.scaling_layer.weight.data = torch.FloatTensor(data=np.eye(input_dim, input_dim))

        # Parameters for adaptive scaling
        self.gating_layer = nn.Linear(input_dim, input_dim)

        self.eps = 1e-8

    def __step1(self, x):
        avg = torch.mean(x, 2)
        adaptive_avg = self.mean_layer(avg)
        adaptive_avg = adaptive_avg.resize(adaptive_avg.size(0), adaptive_avg.size(1), 1)
        x = x - adaptive_avg
        return x

    def __step2(self, x):
        std = torch.mean(x ** 2, 2)
        std = torch.sqrt(std + self.eps)
        adaptive_std = self.scaling_layer(std)
        adaptive_std[adaptive_std <= self.eps] = 1
        adaptive_std = adaptive_std.resize(adaptive_std.size(0), adaptive_std.size(1), 1)
        x = x / adaptive_std
        return x

    def __step3(self, x):
        avg = torch.mean(x, 2)
        gate = F.sigmoid(self.gating_layer(avg))
        gate = gate.resize(gate.size(0), gate.size(1), 1)
        x = x * gate
        return x

    def forward(self, x):
        # Expecting  (n_samples, dim,  n_feature_vectors)

        # Nothing to normalize
        if self.mode == None:
            pass

        # Do simple average normalization
        elif self.mode == 'avg':
            avg = torch.mean(x, 2)
            avg = avg.resize(avg.size(0), avg.size(1), 1)
            x = x - avg

        # Perform only the first step (adaptive averaging)
        elif self.mode == 'adaptive_avg':
            x = self.__step1(x)

        # Perform the first + second step (adaptive averaging + adaptive scaling)
        elif self.mode == 'adaptive_scale':
            x = self.__step1(x)
            x = self.__step2(x)

        # Perform all the three steps
        elif self.mode == 'full':
            x = self.__step1(x)
            x = self.__step2(x)
            x = self.__step3(x)

        else:
            assert False

        return x