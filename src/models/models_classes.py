
# MODELS
import src.models.mlp.mlp as mlp
import src.models.cnn1.cnn1 as cnn1
import src.models.cnn2.cnn2 as cnn2
import src.models.binctabl.binctabl as binctabl

from enum import Enum


class Models(Enum):
    MLP = mlp.MLP_lm
    CNN1 = cnn1.CNN_lm
    CNN2 = cnn2.CNN2_ml
    BINCTABL = binctabl.BinCTABL_ml
