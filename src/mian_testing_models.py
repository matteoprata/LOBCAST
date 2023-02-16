import os
import sys

# preamble needed for cluster
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.main_single import *

# 'model=AXIALLOB-seed=0-trst=FI-test=FI-data=FI-peri=FI-bw=None-fw=None-fiw=10'
kset, mset = cst.FI_Horizons, cst.Models
seed = 0
stock_dataset = "FI"
src_data = "data/saved_models/LOB-CLASSIFIERS-(FI-2010-Sweep-ALL)/"

for k in kset:
    for m in mset:
        fn = "model={}-seed={}-trst={}-test={}-data={}-peri={}-bw={}-fw={}-fiw={}"
        fn = fn.format(m.name, seed, stock_dataset, stock_dataset, stock_dataset, stock_dataset, None, None, k.value)
        aa = src_data + fn

        f1_fname = dict()
        folder = src_data + fn
        for f in os.listdir(folder):
            f1 = float(f.split("validation_FI_f1=")[1].split(".ckpt")[0])
            f1_fname[f] = f1

        max_key, max_value = max(f1_fname.items(), key=lambda x: x[1])
        file = folder + max_key
        