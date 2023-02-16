import os
import sys
import torch

# preamble needed for cluster
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.config import Configuration
import src.constants as cst
from src.main_single import *
from src.main_helper import pick_dataset, pick_model
from src.models.model_executor import NNEngine

kset, mset = cst.FI_Horizons, cst.Models
seed = 0
stock_dataset = "FI"
src_data = "data/saved_models/LOB-CLASSIFIERS-(FI-2010-Sweep-ALL)/"
dirs = [d + '/' for d in os.listdir(src_data) if not d.startswith('.')]


def launch_test(cf: Configuration):
    for k in kset:
        for model in mset:
            cf.CHOSEN_MODEL = model
            cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON] = k.value

            dir_name = "model={}-seed={}-trst={}-test={}-data={}-peri={}-bw={}-fw={}-fiw={}/".format(
                cf.CHOSEN_MODEL.name,
                cf.SEED,
                cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN].name,
                cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST].name,
                cf.CHOSEN_DATASET.value,
                cf.CHOSEN_PERIOD.name,
                cf.HYPER_PARAMETERS[cst.LearningHyperParameter.BACKWARD_WINDOW],
                cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW],
                cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON],
            )

            files =  [f for f in os.listdir(src_data + dir_name) if not f.startswith('.')]
            assert len(files) == 1, 'We expect that in the folder there is only the checkpoint with the highest F1-score'
            file_name = files[0]

            checkpoint_file_path = src_data + dir_name + file_name

            datamodule = pick_dataset(cf)
            model = NNEngine(cf)
            trainer = Trainer(accelerator=cst.DEVICE_TYPE, devices=cst.NUM_GPUS)

            trainer.test(model=model, datamodule=datamodule, ckpt_path=checkpoint_file_path)

            cf.METRICS_JSON.close()

            break



if __name__ == "__main__":

    cf = set_configuration()
    set_seeds(cf)

    cf.CHOSEN_DATASET = cst.DatasetFamily.FI
    cf.CHOSEN_STOCKS[cst.STK_OPEN.TRAIN] = cst.Stocks.FI
    cf.CHOSEN_STOCKS[cst.STK_OPEN.TEST] = cst.Stocks.FI
    cf.CHOSEN_PERIOD = cst.Periods.FI
    cf.IS_WANDB = 0
    cf.IS_TUNE_H_PARAMS = False

    launch_test(cf)


