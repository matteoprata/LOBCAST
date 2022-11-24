
import src.config as co

hyperparameters_cnn2 = {

    # co.TuningVars.EPOCHS.value: {'values': [5, 10, 15]},
    co.TuningVars.OPTIMIZER.value: {'values': [co.Optimizers.RMSPROP.value]},
    co.TuningVars.LEARNING_RATE.value: {'max': 0.001, 'min': 0.0001},

    co.TuningVars.BATCH_SIZE.value: {'values': [16, 32, 64]},  # [32, 64, 128]
    co.TuningVars.IS_SHUFFLE.value: {'values': [True]},

}
