
import src.config as co

hyperparameters_cnn1 = {

    co.TuningVars.EPOCHS.value: {'values': [100]},
    co.TuningVars.OPTIMIZER.value: {'values': [co.Optimizers.ADAM.value]},
    co.TuningVars.LEARNING_RATE.value: {'max': 0.001, 'min': 0.0001},

    co.TuningVars.BATCH_SIZE.value: {'values': [16]},
    co.TuningVars.IS_SHUFFLE.value: {'values': [True]},
    co.TuningVars.NUM_SNAPSHOTS.value: {'values': [100]},

}
