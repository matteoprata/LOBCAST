
import src.constants as cst

hyperparameters_lstm = {

    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'max': 0.001, 'min': 0.0001},  # 'max': 0.001, 'min': 0.0001

    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [32]},  # [32, 64, 128]
    cst.LearningHyperParameter.IS_SHUFFLE_TRAIN_SET.value: {'values': [True]},
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: {'values': [100]},

    cst.LearningHyperParameter.LSTM_HIDDEN.value: {'values': [40]},  # [32, 40, 48]
    cst.LearningHyperParameter.LSTM_N_HIDDEN.value: {'values': [1]},
    cst.LearningHyperParameter.MLP_HIDDEN.value: {'values': [64]},

    cst.LearningHyperParameter.P_DROPOUT.value: {'values': [0]},

}
