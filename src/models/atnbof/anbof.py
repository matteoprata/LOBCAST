import torch
import torch.nn as nn

from src.models.atnbof.layers import ResNetPreprocessing, Attention
from src.models.atnbof.nbof import NBoF


class ANBoF(nn.Module):
    """
    Attention Neural Bag of Feature model
    """

    def __init__(self, in_channels, series_length, n_codeword, att_type, n_class, dropout):
        super(ANBoF, self).__init__()
        # resnet preprocessing block
        self.resnet_block = ResNetPreprocessing(in_channels=in_channels)

        # nbof block
        in_channels, series_length = self.compute_intermediate_dimensions(in_channels, series_length)
        self.quantization_block = NBoF(in_channels, n_codeword)

        # attention block
        self.attention_block = Attention(n_codeword, series_length, att_type)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=n_codeword, out_features=512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=512, out_features=n_class)
        )

    def forward(self, x):
        x = self.resnet_block(x)
        x = self.quantization_block(x).mean(-1)
        x = self.classifier(x)
        return x

    def compute_intermediate_dimensions(self, in_channels, series_length):
        with torch.no_grad():
            x = torch.randn(1, in_channels, series_length)
            y = self.resnet_block(x)
            n_channels = y.size(1)
            length = y.size(2)
            return n_channels, length

    def get_parameters(self):
        bn_params, other_params = self.resnet_block.get_parameters()
        other_params.extend(list(self.quantization_block.parameters()))
        other_params.extend(list(self.attention_block.parameters()))
        other_params.extend(list(self.classifier.parameters()))
        return bn_params, other_params



