
import src.config as co

hyperparameters_lobster = {

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

    co.TuningVars.LABELING_SIGMA_SCALER.value: {
        'values': [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    },

}
