
import src.constants as cst

HP_CNNLSTM = {

    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.RMSPROP.value]},
    cst.LearningHyperParameter.RNN_HIDDEN.value: {'values': [32]},
    cst.LearningHyperParameter.RNN_N_HIDDEN.value: {'values': [1]},
    cst.LearningHyperParameter.MLP_HIDDEN.value: {'values': [32]},
    cst.LearningHyperParameter.P_DROPOUT.value: {'values': [0.1]},
    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [128]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.001]}
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

HP_CNNLSTM_LOBSTER_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.RMSPROP.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.001,
    cst.LearningHyperParameter.BATCH_SIZE.value: 32,
    cst.LearningHyperParameter.RNN_HIDDEN.value: 32,
    cst.LearningHyperParameter.RNN_N_HIDDEN.value: 1,
    cst.LearningHyperParameter.MLP_HIDDEN.value: 32,
    cst.LearningHyperParameter.P_DROPOUT.value: 0.1
}

