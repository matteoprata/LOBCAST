
import src.constants as cst

HP_DEEP = {
    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    cst.LearningHyperParameter.EPS.value: {'values': [1]},
    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [32]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.01]}
}

HP_DEEP_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.01,
    cst.LearningHyperParameter.BATCH_SIZE.value: 32,
    cst.LearningHyperParameter.EPS.value: 1,
}

HP_DEEP_LOBSTER_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.01,
    cst.LearningHyperParameter.BATCH_SIZE.value: 32,
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: 100,
}
