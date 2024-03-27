
import pytorch_lightning as pl
from src.hyper_parameters import HPTunable


class LOBCAST_module:
    def __init__(self, model, tunable_parameters=None):
        self.model = model
        self.tunable_parameters = tunable_parameters if tunable_parameters is not None else HPTunable()
        self.name = model.__class__.__name__
        self.line_color = "red"
        self.line_shape = "-"


class LOBCAST_model(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
