
import src.constants as cst
from enum import Enum
from typing import List, Union
from src.config import Configuration
import src.utils.utils_training_loop as tru
import argparse
from src.utils.utils_dataset import pick_dataset
from src.utils.utils_models import pick_model


class Instance:
    def __init__(self, model: cst.Models, horizons: List[int], seeds: List[int]):
        self.model = model
        self.horizons = horizons
        self.seeds = seeds


# def set_seeds(config: Configuration):
#     """ Sets the random seed to all the random generators. """
#     seed_everything(config.SEED)
#     np.random.seed(config.SEED)
#     random.seed(config.SEED)
#     config.RANDOM_GEN_DATASET = np.random.RandomState(config.SEED)

from pytorch_lightning import Trainer


def run_instance():
    cf: Configuration = Configuration()
    print("created cf")
    parse_cl_arguments(cf)
    print("parsed cf")
    # tru.run(cf)

    # assign parameters of the specific model to the config file
    data_module = pick_dataset(cf)     # load the data
    nn = pick_model(cf, data_module)   # load the model

    trainer = Trainer(
        accelerator=cst.DEVICE_TYPE,
        devices=cst.NUM_GPUS,
        check_val_every_n_epoch=5,
        max_epochs=100,
        callbacks=[
        ],
    )
    trainer.fit(nn, data_module)


def run(instances):
    for instance in instances:
        for seed in instance.seeds:
            for win in instance.horizons:

                print(f"Running LOB experiment: model={instance.model}, fw={win}, seed={seed}")

                try:
                    cf: Configuration = Configuration()
                    cf.SEED = seed
                    cf.PREDICTION_MODEL = instance.model
                    cf.CHOSEN_PERIOD = instance.horizons

                    tru.set_seeds(cf)

                    cf.DATASET_NAME = dataset
                    if mod == cst.Models.METALOB:
                        cf.DATASET_NAME = cst.DatasetFamily.META
                        cf.TARGET_DATASET_META_MODEL = target_dataset_meta
                        cf.JSON_DIRECTORY = json_dir

                    cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks.ALL
                    cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks.ALL
                    cf.CHOSEN_PERIOD = peri

                    cf.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW] = cst.WinSize.EVENTS1.value
                    cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW] = window_forward.value

                    cf.PREDICTION_MODEL = mod

                    cf.IS_WANDB = int(not is_debug)
                    cf.IS_HPARAM_SEARCH = int(not is_debug)

                    tru.run(cf)

                except KeyboardInterrupt:
                    print("There was a problem running on", server_name.name, "LOB experiment on {}, with K+={}".format(mod, window_forward))
                    sys.exit()


def parse_cl_arguments(configuration: Configuration):
    """ Parses the arguments for the command line. """

    parser = argparse.ArgumentParser(description='LOBCAST single execution arguments:')

    parser.add_argument('-seed',   '--SEED', default=configuration.SEED, type=int)
    parser.add_argument('-wf',     '--WINDOW_FUTURE', default=configuration.SEED, type=int)
    parser.add_argument('-wp',     '--WINDOW_PAST', default=configuration.SEED, type=int)
    parser.add_argument('-wunit',  '--WINDOW_UNIT', default=configuration.SEED, type=int)
    parser.add_argument('-model',  '--PREDICTION_MODEL', default=configuration.SEED, type=int)
    parser.add_argument('-search', '--IS_HPARAM_SEARCH', default=configuration.SEED, type=int)
    parser.add_argument('-stock',  '--STOCK', default=configuration.SEED, type=int)
    parser.add_argument('-cpu',    '--IS_CPU', default=configuration.SEED, type=int)

    args = vars(parser.parse_args())

    print("Setting parameters...")

    configuration.SEED = args["SEED"]


if __name__ == '__main__':
    # models = [cst.Models.MLP, cst.Models.CNN1]
    # HORIZONS = [10, 20, 30, 40, 50, 60, 70, 80]
    # SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    #
    # instances = []
    # instances += [Instance(model, HORIZONS, SEEDS) for model in models]
    # instances += [Instance(cst.Models.MLP , [10], [1])]
    # instances += [Instance(cst.Models.CNN2, [10], [1])]

    run_instance()

# ARGUMENTS SINGLE:
#
# model
# horizon +
# horizon -
# seed
# dataset name (whole period, just to split)
# is_hyper_search  (else read parameters from json)
# chosen stock
#
