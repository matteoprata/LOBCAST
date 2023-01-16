
import src.constants as cst

HP_TRANS = {
    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.0001]},  # 'max': 0.001, 'min': 0.0001
    cst.LearningHyperParameter.WEIGHT_DECAY.value: {'values': [1e-5]},
    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [32]},  # [32, 64, 128]
}


HP_TRANS_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.0001,
    cst.LearningHyperParameter.WEIGHT_DECAY.value: 0.00001,
    cst.LearningHyperParameter.BATCH_SIZE.value: 32,
}

# TODO add dropout (hard encoded fn)
