import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, n_filter, kernel_size, padding, dropout, pooling, expand_right):
        super(ResidualBlock, self).__init__()

        left_branch = [
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=in_channels, out_channels=n_filter, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(n_filter),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=n_filter, out_channels=n_filter, kernel_size=kernel_size, padding=padding)
        ]

        right_branch = []
        if pooling:
            left_branch.append(nn.MaxPool1d(kernel_size=2, stride=2))
            if expand_right:
                right_branch.append(nn.Conv1d(in_channels=in_channels, out_channels=n_filter, kernel_size=1, stride=1))
                right_branch.append(nn.MaxPool1d(kernel_size=2, stride=2))
            else:
                right_branch.append(nn.MaxPool1d(kernel_size=2, stride=2))

        else:
            if expand_right:
                right_branch.append(nn.Conv1d(in_channels=in_channels, out_channels=n_filter, kernel_size=1))

        if len(right_branch) == 0:
            self.right_branch = None
        elif len(right_branch) == 1:
            self.right_branch = right_branch[0]
        else:
            self.right_branch = nn.Sequential(*right_branch)

        self.left_branch = nn.Sequential(*left_branch)
        self.initialize()
        self.expand_right = expand_right
        self.pooling = pooling

    def forward(self, x):
        if self.right_branch is not None:
            left = self.left_branch(x)
            right = self.right_branch(x)
            x = left + right
        else:
            x = self.left_branch(x) + x

        return x

    def initialize(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv1d):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0.0)

    def get_parameters(self):
        bn_params = list(self.left_branch[0].parameters()) + list(self.left_branch[4].parameters())
        other_params = list(self.left_branch[3].parameters()) + list(self.left_branch[7].parameters())

        if self.expand_right:
            if self.pooling:
                other_params.extend(list(self.right_branch[0].parameters()))
            else:
                other_params.extend(list(self.right_branch.parameters()))

        return bn_params, other_params


class ResNetPreprocessing(nn.Module):
    def __init__(self, in_channels, n_filter=64, kernel_size=15, padding=7, dropout=0.5):
        super(ResNetPreprocessing, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=n_filter, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(n_filter),
            nn.ReLU()
        )

        self.block2_1 = nn.Sequential(
            nn.Conv1d(in_channels=n_filter, out_channels=n_filter, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(n_filter),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=n_filter, out_channels=n_filter, kernel_size=kernel_size, padding=padding),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.block2_2 = nn.MaxPool1d(kernel_size=2, stride=2)

        residual_blocks = []
        pooling = False
        filter_multiplier = 1
        in_channels = n_filter

        for l in range(15):
            if l % 4 == 0 and l > 0:
                filter_multiplier += 1
                expand_right = True
            else:
                expand_right = False

            residual_blocks.append(
                ResidualBlock(
                    in_channels,
                    n_filter * filter_multiplier,
                    kernel_size,
                    padding,
                    dropout,
                    pooling,
                    expand_right
                )
            )
            pooling = not pooling
            in_channels = n_filter * filter_multiplier

        self.residual_blocks = nn.Sequential(*residual_blocks)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2_1(x) + self.block2_2(x)
        x = self.residual_blocks(x)
        return x

    def get_parameters(self):
        bn_params, other_params = [], []

        # block 1
        bn_params.extend(list(self.block1[1].parameters()))
        other_params.extend(list(self.block1[0].parameters()))

        # block 2_1
        bn_params.extend(list(self.block2_1[1].parameters()))
        other_params.extend(list(self.block2_1[0].parameters()))
        other_params.extend(list(self.block2_1[4].parameters()))

        # residual blocks
        for layer in self.residual_blocks:
            bn, other = layer.get_parameters()
            bn_params.extend(bn)
            other_params.extend(other)

        return bn_params, other_params


class Attention(nn.Module):
    def __init__(self, n_codeword, series_length, att_type):
        super(Attention, self).__init__()

        assert att_type in ['temporal', 'spatial']

        self.att_type = att_type
        self.att_dim = n_codeword if att_type == 'spatial' else series_length
        self.w_a = nn.Parameter(torch.Tensor(self.att_dim, self.att_dim))

        self.alpha = nn.Parameter(torch.Tensor(1))
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
        mask = F.softmax(mask, dim=-1)

        # clip the value of alpha to [0, 1]
        with torch.no_grad():
            self.alpha.copy_(torch.clip(self.alpha, 0.0, 1.0))

        # apply mask to input
        x = x * self.alpha + (1.0 - self.alpha) * x * mask

        if self.att_type == 'spatial':
            x = x.transpose(-1, -2)

        return x
