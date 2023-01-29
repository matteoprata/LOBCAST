# Temporal Logistic Neural Bag-of-Features for Financial Time series Forecasting leveraging Limit Order Book Data
# Source: https://www.sciencedirect.com/science/article/pii/S0167865520302245


import torch
import torch.nn as nn
import torch.nn.functional as F


class TLONBoF(nn.Module):

    def __init__(self, window=50, split_horizon=5, n_codewords=256, n_conv=256, use_scaling=True):
        super(TLONBoF, self).__init__()

        self.split_horizon = split_horizon
        self.window = window
        self.n_codewords = n_codewords
        self.n_levels = int(window / split_horizon)
        self.use_scaling = use_scaling

        self.fc1 = nn.Linear(n_codewords * self.n_levels + n_codewords, 512)
        self.fc2 = nn.Linear(512, 3)

        self.a = nn.Parameter(torch.FloatTensor(data=[1]))
        self.c = nn.Parameter(torch.FloatTensor(data=[0]))

        self.a2 = nn.Parameter(torch.FloatTensor(data=[1]))
        self.c2 = nn.Parameter(torch.FloatTensor(data=[0]))

        self.n1 = nn.Parameter(torch.FloatTensor(data=[self.n_codewords]))
        self.n2 = nn.Parameter(torch.FloatTensor(data=[self.split_horizon]))

        self.n12 = nn.Parameter(torch.FloatTensor(data=[self.n_codewords]))
        self.n22 = nn.Parameter(torch.FloatTensor(data=[self.split_horizon]))

        # Dictionary
        self.bof_conv = nn.Conv1d(n_conv, n_codewords, kernel_size=1)

        # Dictionary 2
        self.bof_conv2 = nn.Conv1d(40, n_codewords, kernel_size=1)

        # Input Convolutional
        self.input_conv = nn.Conv1d(40, n_conv, kernel_size=5, padding=2)
        self.bn_cnv = nn.BatchNorm1d(n_conv)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def apply_temporal_bof_input(self, x):
        # Step 1: Measure the similarity with each codeword
        x = self.bof_conv2(x)

        # Step 2: Scale to ensure that the resulting value encodes the similarity
        if self.use_scaling:
            x = torch.tanh(self.a2.expand_as(x)*x + self.c2.expand_as(x))
        else:
            x = torch.tanh(x)
        x = (x + 1) / 2.0

        # Step 3: Create the similarity vectors for each of the input feature vector
        x = (x / torch.sum(x, dim=1, keepdim=True)) * self.n12
        # Step 4: Perform temporal pooling
        x = F.avg_pool1d(x, 15) * 15

        # Flatten the histogram
        x = x.reshape((x.size(0), -1))

        return x


    def apply_temporal_bof(self, x):
        # Step 1: Measure the similarity with each codeword
        x = self.bof_conv(x)

        # Step 2: Scale to ensure that the resulting value encodes the similarity
        if self.use_scaling:
            x = torch.tanh(self.a.expand_as(x)*x + self.c.expand_as(x))
        else:
            x = torch.tanh(x)
        x = (x + 1) / 2.0

        # Step 3: Create the similarity vectors for each of the input feature vector
        x = (x / torch.sum(x, dim=1, keepdim=True)) * self.n1

        # Step 4: Perform temporal pooling
        x = F.avg_pool1d(x, self.split_horizon) * self.n2

        # Flatten the histogram
        x = x.reshape((x.size(0), -1))

        return x

    def forward(self, x):

        # PyTorch needs the channels first (n_samples, dim,  n_feature_vectors)
        x = x.transpose(1, 2)

        histogram2 = self.apply_temporal_bof_input(x)  # *0
        # Apply a convolutional layer
        x = self.input_conv(x)
        x = torch.tanh(x)

        # Apply a temporal BoF
        temporal_histogram = self.apply_temporal_bof(x)

        temporal_histogram = torch.cat([temporal_histogram, histogram2], dim=1)

        # Classifier
        x = self.relu(self.fc1(temporal_histogram))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
