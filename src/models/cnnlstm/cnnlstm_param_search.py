
import src.constants as cst

HP_CNNLSTM = {

    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.RMSPROP.value]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'max': 0.001, 'min': 0.0001},  # 'max': 0.001, 'min': 0.0001
    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [16, 32, 64]},  # [32, 64, 128]
    cst.LearningHyperParameter.LSTM_HIDDEN.value: {'values': [32]},
    cst.LearningHyperParameter.LSTM_N_HIDDEN.value: {'values': [1]},
    cst.LearningHyperParameter.MLP_HIDDEN.value: {'values': [32, 64]},
    cst.LearningHyperParameter.P_DROPOUT.value: {'values': [0.1]},
}


HP_CNNLSTM_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.RMSPROP.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.01,
    cst.LearningHyperParameter.BATCH_SIZE.value: 32,
    cst.LearningHyperParameter.LSTM_HIDDEN.value: 32,
    cst.LearningHyperParameter.LSTM_N_HIDDEN.value: 1,
    cst.LearningHyperParameter.MLP_HIDDEN.value: 64,
    cst.LearningHyperParameter.P_DROPOUT.value: 0.1
}

