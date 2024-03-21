
import pytorch_lightning as pl


class LOBCAST_module:
    def __init__(self, name, model, tunable_parameters):
        self.name = name
        self.model = model
        self.tunable_parameters = tunable_parameters

        self.line_color = "red"
        self.line_shape = "-"


class LOBCAST_model(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
