

import argparse
import os
from datetime import datetime
from enum import Enum
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
from pytorch_lightning import seed_everything

import src.constants as cst
from src.metrics.metrics_log import Metrics

np.set_printoptions(suppress=True)
from src.utils.utils_generic import str_to_bool
from src.settings import Settings
from src.hyper_parameters import ConfigHPTunable, ConfigHPTuned

from src.models.model_callbacks import callback_save_model
from src.data_preprocessing.utils_dataset import pick_dataset
from src.models.utils_models import pick_model
from pytorch_lightning import Trainer
from src.metrics.report import plot_metric_training, plot_metric_best, saved_metrics
from src.utils.utils_generic import get_class_arguments


class LOBCAST:
    def __init__(self):

        self.SETTINGS = Settings()
        self.TUNABLE_H_PRAM = ConfigHPTunable()
        self.TUNED_H_PRAM = ConfigHPTuned()

    def update_settings(self, setting_params):
        # settings new settings
        for key, value in setting_params.items():
            self.SETTINGS.__setattr__(key, value)

        self.SETTINGS.check_parameters_validity()

        # based on the settings
        self.__init_hyper_parameters()

        # at this point parameters are set
        print("\nRunning with settings:\n", self.SETTINGS.__dict__)

    def update_hyper_parameters(self, tuning_parameters):
        # coming from wandb or from local grid search
        for key, value in tuning_parameters.items():
            self.TUNED_H_PRAM.__setattr__(key, value)

        # at this point parameters are set
        print("\nRunning with hyper parameters:\n", self.TUNED_H_PRAM.__dict__)

    def end_setup(self, wandb_instance=None):
        self.__seed_everything(self.SETTINGS.SEED)

        self.DATE_TIME = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        self.__setup_all_directories(self.DATE_TIME, self.SETTINGS)

        self.METRICS = Metrics(self.SETTINGS.DIR_EXPERIMENTS, self.sim_name_format())
        self.METRICS.dump_info(self.SETTINGS.__dict__, self.TUNED_H_PRAM.__dict__)
        self.WANDB_INSTANCE = wandb_instance

    def __init_hyper_parameters(self):
        model_arguments = get_class_arguments(self.SETTINGS.PREDICTION_MODEL.value.model)[2:]
        model_tunable = self.SETTINGS.PREDICTION_MODEL.value.tunable_parameters

        # checks that HP are meaningful
        for param, values in model_tunable.__dict__.items():
            if not (param in model_arguments or param in self.TUNABLE_H_PRAM.__dict__):
                raise KeyError(f"The declared hyper parameters \'{param}\' of model {self.SETTINGS.PREDICTION_MODEL.name} is never used. Remove it.")

        self.TUNABLE_H_PRAM = model_tunable

        # set to default, add the same parameters in the TUNED_H_PRAM object
        for key, _ in self.TUNABLE_H_PRAM.__dict__.items():
            self.TUNED_H_PRAM.add_hyperparameter(key, None)

    def __seed_everything(self, seed):
        """ Sets the random seed to all the random generators. """
        seed_everything(seed)

    def sim_name_format(self):
        SIM_NAME = "MOD={}-SEED={}-DS={}-HU={}-HP={}-HF={}-OB={}"
        return SIM_NAME.format(
            self.SETTINGS.PREDICTION_MODEL.name,
            self.SETTINGS.SEED,
            self.SETTINGS.DATASET_NAME.value,
            self.SETTINGS.PREDICTION_HORIZON_UNIT.name,
            self.SETTINGS.PREDICTION_HORIZON_PAST,
            self.SETTINGS.PREDICTION_HORIZON_FUTURE,
            self.SETTINGS.OBSERVATION_PERIOD,
        )

    def parse_cl_arguments(self):
        """ Parses the arguments for the command line. """
        parser = argparse.ArgumentParser(description='LOBCAST arguments:')

        # every field in the settings, can be set crom cl
        for k, v in self.SETTINGS.__dict__.items():
            var = v.name if isinstance(v, Enum) else v
            type_var = str if isinstance(v, Enum) else type(v)
            type_var = str_to_bool if type(v) == bool else type_var  # to parse bool
            parser.add_argument(f'--{k}', default=var, type=type_var)

        args = vars(parser.parse_args())

        print("Gathering CLI values.")
        setting_conf = dict()
        # every field in the settings, is set based on the parsed values, enums are parsed by NAME
        for k, v in self.SETTINGS.__dict__.items():
            value = v.__class__[args[k]] if isinstance(v, Enum) else args[k]
            setting_conf[k] = value

        return setting_conf

    @staticmethod
    def __setup_all_directories(fname, settings):
        settings.DIR_EXPERIMENTS = f"{cst.DIR_EXPERIMENTS}-({fname})/"

        # create the paths for the simulation if they do not exist already
        paths = ["data", settings.DIR_EXPERIMENTS]
        for p in paths:
            if not os.path.exists(p):
                os.makedirs(p)

    def run(self):
        """ Given a simulation, settings and hyper params, it runs the training loop. """

        data_module = pick_dataset(self)
        nets_module = pick_model(self, data_module, self.METRICS)


        trainer = Trainer(
            accelerator=self.SETTINGS.DEVICE,
            devices=self.SETTINGS.N_GPUs,
            check_val_every_n_epoch=self.SETTINGS.VALIDATION_EVERY,
            max_epochs=self.SETTINGS.EPOCHS_UB,
            num_sanity_val_steps=0,
            callbacks=[
                callback_save_model(self.SETTINGS.DIR_EXPERIMENTS, self.sim_name_format(), cst.VALIDATION_METRIC, top_k=3)
            ],
        )

        model_path = self.SETTINGS.TEST_MODEL_PATH if self.SETTINGS.IS_TEST_ONLY else "best"

        if not self.SETTINGS.IS_TEST_ONLY:
            trainer.fit(nets_module, data_module)
            self.METRICS.reset_stats()

            # this flag is used when running simulation to know if final validation on best model is running
            self.METRICS.is_best_model = True

            # best model evaluation starts
            trainer.validate(nets_module, data_module, ckpt_path=model_path)
        trainer.test(nets_module, data_module, ckpt_path=model_path)
        self.__plot_stats()
        print('Completed.')

    def __plot_stats(self):
        fnames_root = self.SETTINGS.DIR_EXPERIMENTS + self.sim_name_format()
        pdf_best    = PdfPages(fnames_root + "_" + 'metrics_best_plots.pdf')
        pdf_running = PdfPages(fnames_root + "_" + 'metrics_train_plots.pdf')

        for m in saved_metrics:
            plot_metric_best(fnames_root + "_" + cst.METRICS_BEST_FILE_NAME, m, pdf_best)
            plot_metric_training(fnames_root + "_" + cst.METRICS_RUNNING_FILE_NAME, m, pdf_running)

        pdf_best.close()
        pdf_running.close()
