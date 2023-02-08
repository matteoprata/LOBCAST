import torch.nn as nn


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
        bn_params = list(self.left_branch[0].parameters()) +\
            list(self.left_branch[4].parameters())
        other_params = list(self.left_branch[3].parameters()) +\
            list(self.left_branch[7].parameters())

        if self.expand_right:
            if self.pooling:
                other_params.extend(list(self.right_branch[0].parameters()))
            else:
                other_params.extend(list(self.right_branch.parameters()))

        return bn_params, other_params
