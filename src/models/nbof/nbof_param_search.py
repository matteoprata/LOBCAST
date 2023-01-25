import src.constants as cst

HP_NBoF = {
    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'max': 0.01, 'min': 0.0001}, # 0.01 in the paper
    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [256]},  # [32, 64, 128]
    cst.LearningHyperParameter.RNN_HIDDEN.value: {'values': [100]},  # [32, 40, 48]

}

HP_NBoF_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.01,
    cst.LearningHyperParameter.BATCH_SIZE.value: 256,
}

HP_NBoF_LOBSTER_FIXED = {

}
