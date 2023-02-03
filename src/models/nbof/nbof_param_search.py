import src.constants as cst

HP_NBoF = {
    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'max': 0.001, 'min': 0.00001}, # 0.01 in the paper
    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [32, 64, 128]},  # [32, 64, 128]
    cst.LearningHyperParameter.MLP_HIDDEN.value: {'values': [512]},
    cst.LearningHyperParameter.NUM_RBF_NEURONS.value: {'values': [16]},
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: {'values': [15]},
}

HP_NBoF_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.0002154,
    cst.LearningHyperParameter.BATCH_SIZE.value: 128,
    cst.LearningHyperParameter.MLP_HIDDEN.value: 512,
    cst.LearningHyperParameter.NUM_RBF_NEURONS.value: 16,
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: 15
}

HP_NBoF_LOBSTER_FIXED = {

}
