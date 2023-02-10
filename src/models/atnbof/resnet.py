import torch.nn as nn
from src.models.atnbof.residual_block import ResidualBlock


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

        for layer_idx in range(15):
            if layer_idx % 4 == 0 and layer_idx > 0:
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
