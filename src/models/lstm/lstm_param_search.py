
import src.constants as cst

HP_LSTM = {
    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    cst.LearningHyperParameter.RNN_HIDDEN.value: {'values': [40]},  # [32, 40, 48]
    cst.LearningHyperParameter.RNN_N_HIDDEN.value: {'values': [1]},
    cst.LearningHyperParameter.MLP_HIDDEN.value: {'values': [64]},
    cst.LearningHyperParameter.P_DROPOUT.value: {'values': [0]},

    cst.LearningHyperParameter.BATCH_SIZE.value:    {'values': [32, 64, 128]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [1e-2, 1e-3, 1e-4]}

}

HP_LSTM_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.0007426,
    # cst.LearningHyperParameter.BATCH_SIZE.value: 64,
    cst.LearningHyperParameter.MLP_HIDDEN.value: 64,
    cst.LearningHyperParameter.RNN_HIDDEN.value: 40,
    cst.LearningHyperParameter.RNN_N_HIDDEN.value: 1,
}
