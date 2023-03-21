import src.constants as cst

HP_META = {
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.SGD.value]},
    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [32]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [.001]},
    cst.LearningHyperParameter.MLP_HIDDEN.value: {'values': [16]},
}


HP_META_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 500,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.SGD.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.01,
    cst.LearningHyperParameter.BATCH_SIZE.value: 32,
    cst.LearningHyperParameter.MLP_HIDDEN: 16
}
