
import src.config as co

hyperparameters_mlp = {

    # co.TuningVars.EPOCHS.value: {'values': [5, 10, 15]},
    co.TuningVars.OPTIMIZER.value: {'values': [co.Optimizers.ADAM.value]},
    co.TuningVars.LEARNING_RATE.value: {'max': 0.001, 'min': 0.0001},
    co.TuningVars.BATCH_SIZE.value: {'values': [16, 32, 48]},       # [32, 64, 128]
    co.TuningVars.IS_SHUFFLE.value: {'values': [True]},

    co.TuningVars.MLP_HIDDEN.value: {'values': [128]}, # [128, 192, 256]

    co.TuningVars.P_DROPOUT.value: {'values': [0]}, # [0.1, 0.25, 0.5]

}
