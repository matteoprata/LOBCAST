
import src.constants as cst

HP_MLP = {
    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'max': 0.001, 'min': 0.0001},
    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [16, 32, 48]},       # [32, 64, 128]
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: {'values': [100]},
    cst.LearningHyperParameter.MLP_HIDDEN.value: {'values': [128, 192, 256]},  # [128, 192, 256]
    cst.LearningHyperParameter.P_DROPOUT.value: {'values': [0]},  # [0.1, 0.25, 0.5]
}


HP_MLP_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.0004883,
    cst.LearningHyperParameter.BATCH_SIZE.value: 16,
    cst.LearningHyperParameter.MLP_HIDDEN.value: 256,
}
