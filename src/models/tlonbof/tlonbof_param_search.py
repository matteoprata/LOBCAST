import src.constants as cst

HP_TLONBoF = {
    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: {'values': [15]},

    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [32, 64, 128]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.0001, 0.000325, 0.00055, 0.000775, 0.001]}, # {'max': 0.01, 'min': 0.001},  # 'max': 0.001, 'min': 0.0001

}

HP_TLONBoF_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 1e-4,
    cst.LearningHyperParameter.BATCH_SIZE.value: 128,
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: 15
}

HP_TLONBoF_LOBSTER_FIXED = {

}
