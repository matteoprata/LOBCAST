import torch
import torch.nn as nn
import torch.nn.functional as F


class NBoF(nn.Module):
    def __init__(self, in_channels, n_codeword):
        super(NBoF, self).__init__()

        self.codebook = nn.Parameter(
            data=torch.Tensor(n_codeword, in_channels),
            requires_grad=True
        )
        self.scaling_a = nn.Parameter(data=torch.Tensor(1), requires_grad=True)
        self.scaling_b = nn.Parameter(data=torch.Tensor(1), requires_grad=True)

        nn.init.kaiming_uniform_(self.codebook)
        nn.init.constant_(self.scaling_a, 1.0)
        nn.init.constant_(self.scaling_b, 0.0)

    def forward(self, x):
        # input shape should be: batch_size x in_channels x series_length
        x = x.transpose(-1, -2)

        # dot product similarity
        similarity = F.linear(x, self.codebook)

        # scale and shift before tanh
        similarity = torch.tanh(self.scaling_a * similarity + self.scaling_b)
        similarity = (1.0 + similarity) / 2.0

        # transpose back to correct axis arrangement: batch_size x nb_codewords x series_length
        similarity = similarity.transpose(-1, -2)

        return similarity

