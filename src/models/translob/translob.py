# Transformers for limit order books
# Source: https://arxiv.org/pdf/2003.00130.pdf

import pytorch_lightning as pl
import torch
import torch.nn as nn
import src.config as co


class TransLob(pl.LightningModule):

    def __init__(self):
        super().__init__()

        # self.transformer = nn.Transformer(d_model=15, nhead=3,
        # num_encoder_layers=2, num_decoder_layers=2,
        # dim_feedforward=3000, dropout=0.0, norm_first=True,
        # batch_first=True, device=device)

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=40, out_channels=14, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=14, out_channels=14, kernel_size=2, dilation=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=14, out_channels=14, kernel_size=2, dilation=4, padding=4),
            nn.ReLU(),
            nn.Conv1d(in_channels=14, out_channels=14, kernel_size=2, dilation=8, padding=8),
            nn.ReLU(),
            nn.Conv1d(in_channels=14, out_channels=14, kernel_size=2, dilation=16, padding=16),
            nn.ReLU(),
            nn.BatchNorm1d(14),
        )

        self.dropout = nn.Dropout(0.1)

        self.activation = nn.ReLU()

        self.encoder_layer = nn.TransformerEncoderLayer(15, 3, 60, 0.0, batch_first=True)
        self.norm = nn.BatchNorm1d(100)

        self.transformer = nn.TransformerEncoder(self.encoder_layer, 2, norm=self.norm)

        self.fc1 = nn.Linear(1500, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        '''
        np_tensor = x.cpu().numpy()
        tf_tensor = tf.convert_to_tensor(np_tensor)

        x = lob_dilated(tf_tensor)
        '''

        x = torch.permute(x, (0, 2, 1))
        x = self.conv(x)
        x = x[:, :, :-31]
        x = x.permute(0, 2, 1)
        # np_tensor = x.cpu().detach().numpy()
        # tf_tensor = tf.convert_to_tensor(np_tensor)

        x = self.positional_encoding(x)

        # np_tensor = x.cpu().numpy()
        # np_tensor = np.squeeze(np_tensor)

        # x = torch.tensor(np_tensor).to(device)

        x = self.transformer(x)
        # print(x.shape)
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))

        x = self.dropout(self.activation(self.fc1(x)))
        # x = self.dropout(self.norm2(self.activation(self.fc2(x))))
        x = self.fc2(x)

        return x

    def positional_encoding(self, x):
        n_levels = 100
        pos = torch.arange(0, n_levels, 1, dtype=torch.float32) / (n_levels - 1)
        pos = (pos + pos) - 1
        # pos = np.reshape(pos, (pos.shape[0]))
        pos_final = torch.zeros((x.shape[0], n_levels, 1), dtype=torch.float32, device=co.DEVICE_TYPE)
        for i in range(pos_final.shape[0]):
            for j in range(pos_final.shape[1]):
                pos_final[i, j, 0] = pos[j]

        x = torch.cat((x, pos_final), 2)

        return x
