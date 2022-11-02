
import src.config as co

sweep_configuration_mlpFI = {
    'method': 'bayes',
    'name': 'MLP_150epochs_datasetFI_earlyStopping25',

    'metric': {
        'goal': 'maximize',
        'name': co.ModelSteps.VALIDATION.value + co.Metrics.F1.value
    },

    'parameters': {
        co.TuningVars.LEARNING_RATE.value: {'max': 0.001, 'min': 0.0001},
        co.TuningVars.BATCH_SIZE.value: {'values': [32]},       # [32, 64, 128]
        co.TuningVars.IS_SHUFFLE.value: {'values': [True]},

        co.TuningVars.MLP_HIDDEN.value: {'values': [128, 192, 256]},

        co.TuningVars.FI_HORIZON.value: {'values': [
            co.Horizons.K1.value,
            co.Horizons.K2.value,
            co.Horizons.K3.value,
            co.Horizons.K5.value,
            co.Horizons.K10.value
        ]},

        co.TuningVars.BACKWARD_WINDOW.value: {'values':[
            co.WinSize.SEC10.value,
            co.WinSize.SEC20.value,
            co.WinSize.SEC30.value,
            co.WinSize.SEC50.value,
            co.WinSize.SEC100.value,
        ]}

    }
}
