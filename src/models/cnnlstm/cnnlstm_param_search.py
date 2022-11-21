
import src.config as co

hyperparameters_cnnlstm = {

    # co.TuningVars.EPOCHS.value: {'values': [5, 10, 15]},
    co.TuningVars.LEARNING_RATE.value: {'max': 0.001, 'min': 0.0001},  # 'max': 0.001, 'min': 0.0001
    co.TuningVars.BATCH_SIZE.value: {'values': [16]},  # [32, 64, 128]
    co.TuningVars.IS_SHUFFLE.value: {'values': [True]},

    co.TuningVars.LSTM_HIDDEN.value: {'values': [32]},  # [32, 40, 48]
    co.TuningVars.LSTM_N_HIDDEN.value: {'values': [1]},

    co.TuningVars.P_DROPOUT.value: {'values': [0.1, 0.25, 0.5]},

}
