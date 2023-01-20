# Multi-Horizon Forecasting for Limit Order Books: Novel Deep Learning Approaches and Hardware Acceleration using Intelligent Processing Units
# Source: https://arxiv.org/pdf/2105.10430.pdf

import pytorch_lightning as pl
from torch import nn
import torch
import src.constants as cst


class DeepLobAtt(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # convolution blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
        )

        # inception moduels
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), padding='same'),
            nn.LeakyReLU(negative_slope=0.01),
        )

        # lstm layers
        self.lstm = nn.LSTM(input_size=192, hidden_size=64, num_layers=1, batch_first=True)

        self.decoder_lstm = nn.LSTM(input_size=67, hidden_size=64, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(128, 3)
        self.BN = nn.BatchNorm1d(1, momentum=0.6)

    def forward(self, x):
        x = x[:, None, :, :]  # none stands for the channel

        decoder_inputs = torch.zeros(x.shape[0], 1, 3, device=cst.DEVICE_TYPE)
        decoder_inputs[:, 0, 0] = 1

        h0 = torch.zeros(1, x.size(0), 64, device=cst.DEVICE_TYPE)
        c0 = torch.zeros(1, x.size(0), 64, device=cst.DEVICE_TYPE)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)

        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, (-1, x.shape[1], x.shape[2]))

        encoder_outputs, (h0, c0) = self.lstm(x, (h0, c0))

        states = (h0, c0)

        encoder_state_h = h0.permute(1, 0, 2)

        inputs = torch.concatenate([decoder_inputs, encoder_state_h], dim=2)

        all_outputs = torch.zeros(5, x.shape[0], 3, device=cst.DEVICE_TYPE)

        # we iterate for every horizon (10, 20, 30, 50, 100)
        for i in range(5):
            # we pass in input to the decoder the context vector, the last decoder's output and the last decoder's hidden state
            output, (state_h, state_c) = self.decoder_lstm(inputs, states)

            # computing the attention for the next time step
            attention = torch.bmm(output, encoder_outputs.permute(0, 2, 1))
            attention = torch.softmax(attention, dim=2)

            # computing the context vector
            c_v = torch.bmm(attention, encoder_outputs)
            c_v = self.BN(c_v)

            # creating the input to compute the distribution for the output (3)
            last_inputs = torch.concatenate([c_v, output], dim=2)

            # computing the distribution for the output with the context vector (encoder_outputs) and the decoder's output
            output = self.fc1(last_inputs)
            output = torch.softmax(output, dim=2)

            all_outputs[i] = torch.squeeze(output)
            inputs = torch.concatenate([output, c_v], dim=2)
            states = [state_h, state_c]

        all_outputs = torch.permute(all_outputs, (1, 2, 0))

        # all_outputs.shape = (batch_size, num_classes, num_horizons)
        return all_outputs
