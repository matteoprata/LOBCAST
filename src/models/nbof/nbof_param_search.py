import src.constants as cst

HP_NBoF = {
    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    cst.LearningHyperParameter.MLP_HIDDEN.value: {'values': [512]},
    cst.LearningHyperParameter.NUM_RBF_NEURONS.value: {'values': [16]},
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: {'values': [15]},

    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [32]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.0001, 0.000325, 0.00055, 0.000775, 0.001]}, # {'max': 0.01, 'min': 0.001},  # 'max': 0.001, 'min': 0.0001
}

HP_NBoF_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.001,
    cst.LearningHyperParameter.BATCH_SIZE.value: 32,
    cst.LearningHyperParameter.MLP_HIDDEN.value: 512,
    cst.LearningHyperParameter.NUM_RBF_NEURONS.value: 16,
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: 15
}

HP_NBoF_LOBSTER_FIXED = {

}
