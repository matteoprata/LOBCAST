import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, n_codeword, series_length, att_type):
        super(Attention, self).__init__()

        self.att_type = att_type
        self.att_dim = n_codeword if att_type == 'spatial' else series_length
        self.w_a = nn.Parameter(
            data=torch.Tensor(self.att_dim, self.att_dim),
            requires_grad=True
        )

        self.softmax = nn.Softmax(dim=-1)

        self.alpha = nn.Parameter(
            data=torch.Tensor(1),
            requires_grad=True
        )

        self.register_buffer('I', torch.eye(self.att_dim))
        nn.init.constant_(self.w_a, 1.0 / self.att_dim)
        nn.init.constant_(self.alpha, 0.5)

    def forward(self, x):
        # dimension order of x: batch_size, in_channels, series_length
        if self.att_type == 'spatial':
            x = x.transpose(-1, -2)

        # make weights on the diagonal fixed
        W = self.w_a - self.w_a * self.I + self.I / self.att_dim

        # compute the attention mask
        mask = F.linear(x, W)
        mask = self.softmax(mask)

        # clip the value of alpha to [0, 1]
        with torch.no_grad():
            self.alpha.copy_(torch.clip(self.alpha, 0.0, 1.0))

        # apply mask to input
        x = x * self.alpha + (1.0 - self.alpha) * x * mask

        if self.att_type == 'spatial':
            x = x.transpose(-1, -2)

        return x
