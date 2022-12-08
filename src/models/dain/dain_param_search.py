
import src.config as co

hyperparameters_dain = {

    co.TuningVars.EPOCHS.value: {'values': [100]},
    co.TuningVars.OPTIMIZER.value: {'values': [co.Optimizers.RMSPROP.value]},
    co.TuningVars.LEARNING_RATE.value: {'values': [0.0001]},

    co.TuningVars.IS_SHUFFLE.value: {'values': [True]},
    co.TuningVars.BATCH_SIZE.value: {'values': [16, 32, 64]},
    co.TuningVars.NUM_SNAPSHOTS.value: {'values': [100]},

    co.TuningVars.DAIN_LAYER_MODE.value: {'values': [
        # None,
        # 'avg',
        # 'adaptive_avg',
        # 'adaptive_scale',
        'full'
    ]},

    co.TuningVars.MLP_HIDDEN.value: {'values': [512]},
    co.TuningVars.P_DROPOUT.value: {'values': [0.5]}

}
