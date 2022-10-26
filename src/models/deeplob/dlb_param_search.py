
import src.config as co

sweep_configuration_dlb = {
    'method': 'bayes',
    'name': 'sweep',

    'metric': {
        'goal': 'maximize',
        'name': co.ModelSteps.VALIDATION.value + co.Metrics.F1.value
    },

    'parameters': {
        co.TuningVars.BACKWARD_WINDOW.value: {
            'values': [
                co.WinSize.SEC10.value,
                co.WinSize.SEC20.value,
                co.WinSize.SEC30.value,
                co.WinSize.SEC50.value,
                co.WinSize.SEC100.value
            ]
        },

        co.TuningVars.FORWARD_WINDOW.value: {
            'values': [
                co.WinSize.SEC10.value,
                co.WinSize.SEC20.value,
                co.WinSize.SEC30.value,
                co.WinSize.SEC50.value,
                co.WinSize.SEC100.value
            ]
        },

        # co.TuningVars.LABELING_THRESHOLD.value: {'min': 0.0005, 'max': 0.005},
        co.TuningVars.LABELING_SIGMA_SCALER.value: {'values': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},

        # co.TuningVars.EPOCHS.value:           {'values': [5, 10, 15]},
        # co.TuningVars.LEARNING_RATE.value:      {'max': 0.001, 'min': 0.0001},  # 'max': 0.001, 'min': 0.0001
        co.TuningVars.BATCH_SIZE.value:         {'values': [32, 68, 128]}, # [32, 64, 128]
        co.TuningVars.MLP_HIDDEN.value:         {'values': [200]}, # [150, 175, 200]
        co.TuningVars.IS_SHUFFLE.value:         {'values': [True]}

    }
}
