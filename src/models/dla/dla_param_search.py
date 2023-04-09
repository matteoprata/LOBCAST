import src.constants as cst

HP_DLA = {
    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    cst.LearningHyperParameter.RNN_HIDDEN.value: {'values': [100]},  # [32, 40, 48]
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: {'values': [5]},

    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [32, 64]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [1e-3, 1e-4, 1e-5]}

}

HP_DLA_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.002795,
    # cst.LearningHyperParameter.BATCH_SIZE.value: 64,
    cst.LearningHyperParameter.RNN_HIDDEN.value: 100,
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: 5,
}

HP_DLA_LOBSTER_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.01,
    cst.LearningHyperParameter.BATCH_SIZE.value: 256,
    cst.LearningHyperParameter.RNN_HIDDEN.value: 100,
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: 5,
}
