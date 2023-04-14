
import src.constants as cst

HP_MLP = {
    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: {'values': [100]},
    cst.LearningHyperParameter.MLP_HIDDEN.value: {'values': [256]},  # [128, 192, 256]
    cst.LearningHyperParameter.P_DROPOUT.value: {'values': [0]},  # [0.1, 0.25, 0.5]

    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [64]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [1e-5]}

}

HP_MLP_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 10,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.0004883,
    cst.LearningHyperParameter.BATCH_SIZE.value: 64,
    cst.LearningHyperParameter.MLP_HIDDEN.value: 256,
}

HP_MLP_LOBSTER_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.001,
    cst.LearningHyperParameter.BATCH_SIZE.value: 64,
    cst.LearningHyperParameter.MLP_HIDDEN.value: 256,
}
