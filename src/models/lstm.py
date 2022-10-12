import pytorch_lightning as pl
import torch
from torch import nn

class LSTM(pl.LightningModule):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super().__init__()

        self.num_classes = num_classes # 3
        self.num_layers = num_layers # 1
        self.input_size = input_size # nfeat
        self.hidden_size = hidden_size # 32
        # self.seq_length = seq_length # horizon

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) # lstm
        
        self.fc_1 =  nn.Linear(hidden_size, 64) # fully connected 64 neurons
        self.dropout = nn.Dropout(p=0.2) # not specified 
        self.prelu = nn.PReLU()
        
        self.fc = nn.Linear(64, num_classes) # out layer
    
    def forward(self,x):

        output, (hn, cn) = self.lstm(x) # lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) # reshaping the data for Dense layer next
        
        out = self.fc_1(hn)
        out = self.dropout(out)
        out = self.prelu(out) 
        out = self.fc(out)
        
        return out
 