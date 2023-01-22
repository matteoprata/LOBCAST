
import src.constants as cst

HP_CNN1 = {

    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'max': 0.001, 'min': 0.0001},
    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [16, 32, 64]},
}

# DONE FIXED 21-01-2023
HP_CNN1_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.0004761,
    cst.LearningHyperParameter.BATCH_SIZE.value: 64,
}

