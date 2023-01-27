from src.models.atnbof.nbof import NBoF


class TNBoF(NBoF):
    def forward(self, x):
        split_index = int(x.size(-1) / 2)
        x_short = super(TNBoF, self).forward(x[:, :, split_index:])
        x_long = super(TNBoF, self).forward(x)
        return x_short, x_long
