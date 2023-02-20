import src.constants as cst

HP_META_FI = {
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value,
                                                            cst.Optimizers.SGD.value]},
    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [32, 64, 128]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'max': 0.01, 'min': 0.0001},
    cst.LearningHyperParameter.META_HIDDEN.value: {'values': [16, 24, 32]},
}


HP_META_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 500,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.SGD.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.01,
    cst.LearningHyperParameter.BATCH_SIZE.value: 32,
    cst.LearningHyperParameter.META_HIDDEN: 16
}
