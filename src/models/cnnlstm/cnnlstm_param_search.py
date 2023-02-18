
import src.constants as cst

HP_CNNLSTM = {

    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.RMSPROP.value]},
    cst.LearningHyperParameter.RNN_HIDDEN.value: {'values': [32]},
    cst.LearningHyperParameter.RNN_N_HIDDEN.value: {'values': [1]},
    cst.LearningHyperParameter.MLP_HIDDEN.value: {'values': [32]},
    cst.LearningHyperParameter.P_DROPOUT.value: {'values': [0.1]},

    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.001, 0.0055, 0.01]}, # {'max': 0.01, 'min': 0.001},  # 'max': 0.001, 'min': 0.0001
    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [64]},  # [32, 64, 128]
}

HP_CNNLSTM_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.RMSPROP.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.000306,
    # cst.LearningHyperParameter.BATCH_SIZE.value: 16,
    cst.LearningHyperParameter.RNN_HIDDEN.value: 32,
    cst.LearningHyperParameter.RNN_N_HIDDEN.value: 1,
    cst.LearningHyperParameter.MLP_HIDDEN.value: 32,
    cst.LearningHyperParameter.P_DROPOUT.value: 0.1
}

