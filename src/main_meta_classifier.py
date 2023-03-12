

import os
import sys

# preamble needed for cluster
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.main_single import *


def experiment_FI(kset):

    for k in kset:
        try:
            cf: Configuration = Configuration(now)
            set_seeds(cf)
            parser_cl_arguments(cf)
            cf.CHOSEN_DATASET = cst.DatasetFamily.META
            cf.CHOSEN_MODEL = cst.Models.METALOB

            cf.IS_WANDB = 1
            cf.IS_TUNE_H_PARAMS = True
            cf.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON] = k.value

            #launch_wandb(cf)
            launch_single(cf, model_params=None)
        except KeyboardInterrupt:
            print("There was a problem running experiment with K={}".format(k))
            sys.exit()


kset = [cst.FI_Horizons.K10]

now = "FI-2010-META"
experiment_FI(kset)
