from torch import nn


class MLP(nn.Module):

    def __init__(
            self,
            x_shape,
            num_classes,
            hidden_layer_dim=128,
            p_dropout=.1
    ):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(x_shape, hidden_layer_dim)
        self.leakyReLU = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=p_dropout)
        self.linear2 = nn.Linear(hidden_layer_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # [batch_size x 40 x window]
        x = x.view(x.size(0), -1).float()
        print(x.device)
        exit()
        out = self.linear1(x)
        out = self.leakyReLU(out)
        out = self.dropout(out)

        out = self.linear2(out)

        logits = self.softmax(out)

        return logits