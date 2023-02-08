# Attention-Based Neural Bag-of-Features Learning for Sequence Data
# Source: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9762322

import torch
import torch.nn as nn
from src.models.atnbof.resnet import ResNetPreprocessing
from src.models.atnbof.selfattention import SelfAttention
from src.models.atnbof.attention import Attention
from src.models.atnbof.tnbof import TNBoF


class ATNBoF(nn.Module):
    def __init__(self, in_channels, series_length, n_codeword, att_type, n_class, dropout):
        super(ATNBoF, self).__init__()
        # resnet preprocessing block
        self.resnet_block = ResNetPreprocessing(in_channels=in_channels)

        # tnbof block
        in_channels, series_length = self.compute_intermediate_dimensions(in_channels, series_length)
        self.quantization_block = TNBoF(in_channels, n_codeword)
        out_dim = n_codeword * 2

        # attention block
        self.att_type = att_type
        if att_type in ['spatiotemporal', 'spatialsa', 'temporalsa']:
            out_dim = out_dim * 3
            self.short_attention_block = SelfAttention(n_codeword, series_length - int(series_length / 2), att_type)
            self.long_attention_block = SelfAttention(n_codeword, series_length, att_type)
            self.short_attention_block2 = SelfAttention(n_codeword, series_length - int(series_length / 2), att_type)
            self.long_attention_block2 = SelfAttention(n_codeword, series_length, att_type)
            self.short_attention_block3 = SelfAttention(n_codeword, series_length - int(series_length / 2), att_type)
            self.long_attention_block3 = SelfAttention(n_codeword, series_length, att_type)
        else:
            self.short_attention_block = Attention(n_codeword, series_length - int(series_length / 2), att_type)
            self.long_attention_block = Attention(n_codeword, series_length, att_type)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=out_dim, out_features=512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=512, out_features=n_class)
        )

    def forward(self, x):
        x = x[:, None, :, :]
        x = torch.flatten(x, start_dim=2)
        x = self.resnet_block(x)
        x_short, x_long = self.quantization_block(x)
        if self.att_type in ['spatialsa', 'temporalsa', 'spatiotemporal']:
            x_short1 = self.short_attention_block(x_short)
            x_long1 = self.long_attention_block(x_long)
            x_short2 = self.short_attention_block2(x_short)
            x_long2 = self.long_attention_block2(x_long)
            x_short3 = self.short_attention_block3(x_short)
            x_long3 = self.long_attention_block3(x_long)
            x_short = torch.cat([x_short1, x_short2, x_short3], dim=1)
            x_long = torch.cat([x_long1, x_long2, x_long3], dim=1)
        else:
            x_short = self.short_attention_block(x_short)
            x_long = self.long_attention_block(x_long)
        x_short = x_short.mean(-1)
        x_long = x_long.mean(-1)
        x = torch.cat([x_short, x_long], dim=-1)
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
        other_params.extend(list(self.short_attention_block.parameters()))
        other_params.extend(list(self.long_attention_block.parameters()))
        other_params.extend(list(self.classifier.parameters()))
        return bn_params, other_params
