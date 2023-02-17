
import src.constants as cst

HP_CNN1 = {

    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},

    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.001, 0.00325, 0.0055, 0.00775, 0.01]},  # {'max': 0.01, 'min': 0.001},  # 'max': 0.001, 'min': 0.0001
    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [16, 32, 64]},
}

HP_CNN1_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.000981,
    # cst.LearningHyperParameter.BATCH_SIZE.value: 64,
}

