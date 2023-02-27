
import src.constants as cst

HP_CNN2 = {

    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.RMSPROP.value]},

    cst.LearningHyperParameter.BATCH_SIZE.value:    {'values': [32, 64, 128]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'max': 1e-2, 'min': 1e-5}

}

HP_CNN2_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.RMSPROP.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.0002685,
    # cst.LearningHyperParameter.BATCH_SIZE.value: 32,
}
