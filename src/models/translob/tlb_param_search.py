
import src.constants as cst

HP_TRANS = {
    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    cst.LearningHyperParameter.WEIGHT_DECAY.value: {'values': [0]},

    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [128]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.001]}
}

HP_TRANS_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.0007931,
    cst.LearningHyperParameter.WEIGHT_DECAY.value: 0.0000,
    # cst.LearningHyperParameter.BATCH_SIZE.value: 64,
}

HP_TRANS_LOBSTER_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.0001,
    cst.LearningHyperParameter.WEIGHT_DECAY.value: 0.0000,
    cst.LearningHyperParameter.BATCH_SIZE.value: 32,
}