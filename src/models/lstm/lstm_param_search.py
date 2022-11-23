
import src.config as co

hyperparameters_lstm = {

    # co.TuningVars.EPOCHS.value: {'values': [5, 10, 15]},
    co.TuningVars.LEARNING_RATE.value: {'max': 0.001, 'min': 0.0001},  # 'max': 0.001, 'min': 0.0001
    co.TuningVars.BATCH_SIZE.value: {'values': [32]}, # [32, 64, 128]
    co.TuningVars.IS_SHUFFLE.value: {'values': [True]},

    co.TuningVars.LSTM_HIDDEN.value: {'values': [40]}, # [32, 40, 48]
    co.TuningVars.LSTM_N_HIDDEN.value: {'values': [1]},
    co.TuningVars.MLP_HIDDEN.value: {'values': [64]},

    co.TuningVars.P_DROPOUT.value: {'values': [0]},

}
