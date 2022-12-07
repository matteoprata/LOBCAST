
import src.config as co

hyperparameters_tabl = {

    co.TuningVars.EPOCHS.value: {'values': [200]},

    co.TuningVars.OPTIMIZER.value: {'values': [co.Optimizers.ADAM.value]},
    co.TuningVars.LEARNING_RATE.value: {'values': [0.01]},  # 'max': 0.001, 'min': 0.0001

    co.TuningVars.BATCH_SIZE.value: {'values': [256]}, # [32, 64, 128]
    co.TuningVars.IS_SHUFFLE.value: {'values': [True]},

}