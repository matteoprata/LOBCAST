
import pytorch_lightning as pl


class LOBCAST_module:
    def __init__(self, name, model, tunable_parameters: dict):
        self.name = name
        self.model = model
        self.tunable_parameters = tunable_parameters

        self.line_color = "red"
        self.line_shape = "-"

    # def __check_parameters(self):
    #     list_arguments = utils_generic.get_class_arguments(self.model)
        # arguments = set(list_arguments)
        # tunable = set(self.tunable_parameters.keys())

        # is_sub = tunable.issubset(arguments)
        # excess = tunable - arguments
        # if not is_sub:
        #     raise ValueError(f"\nThe following hps you are trying to tune, are not arguments of your model."
        #                      f" Make sure that arguments in tunable_parameters are a subset of those in your model.\n"
        #                      f" > Arguments in your model: {arguments}\n"
        #                      f" > Argument to optimize: {excess}")
        # else:
        #     print(f"OK. Tuning hps {self.tunable_parameters}")


class LOBCAST_model(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
