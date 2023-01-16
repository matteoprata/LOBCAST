
import src.constants as cst

HP_CNN2 = {

    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.RMSPROP.value]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'max': 0.001, 'min': 0.0001},

    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [16, 32, 64]},  # [32, 64, 128]

}


HP_CNN2_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.RMSPROP.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.01,
    cst.LearningHyperParameter.BATCH_SIZE.value: 32,
}
