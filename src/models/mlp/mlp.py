from torch import nn

class MLPModel(nn.Module):

    def __init__(
            self,
            x_shape,
            y_shape,
            hidden_layer_dim=128
    ):
        super(MLPModel, self).__init__()

        self.linear1 = nn.Linear(x_shape, hidden_layer_dim)
        self.leakyReLU = nn.LeakyReLU()
        self.linear2 = nn.Linear(hidden_layer_dim, y_shape)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.leakyReLU(out)
        out = self.linear2(out)
        logits = self.softmax(out)

        return logits