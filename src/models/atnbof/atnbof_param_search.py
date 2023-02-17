import src.constants as cst

HP_ATNBoF = {
    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [80]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    cst.LearningHyperParameter.WEIGHT_DECAY.value: {'values': [0.0001]},  # 0.0001 in the paper
    cst.LearningHyperParameter.MLP_HIDDEN.value: {'values': [512]},
    cst.LearningHyperParameter.NUM_RBF_NEURONS.value: {'values': [16]},
    cst.LearningHyperParameter.P_DROPOUT.value: {'values': [0.2]},

    cst.LearningHyperParameter.BATCH_SIZE.value:    {'values': [64]},  # [32, 64, 128]
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.0001, 0.00505, 0.01]}  # {'max': 0.01, 'min': 0.0001},  # 0.001 in the paper
}

HP_ATNBoF_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 80,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.001,
    cst.LearningHyperParameter.WEIGHT_DECAY.value: 0.0001,
    cst.LearningHyperParameter.BATCH_SIZE.value: 64,
    cst.LearningHyperParameter.MLP_HIDDEN.value: 512,
    cst.LearningHyperParameter.NUM_RBF_NEURONS.value: 16,
    cst.LearningHyperParameter.P_DROPOUT.value: 0.2
}

HP_ATNBoF_LOBSTER_FIXED = {

}
