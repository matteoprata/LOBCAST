
import src.config as co

hyperparameters_fi = {

    co.TuningVars.FI_HORIZON.value: {'values': [
        co.Horizons.K1.value,
        co.Horizons.K2.value,
        co.Horizons.K3.value,
        co.Horizons.K5.value,
        co.Horizons.K10.value
    ]},

}
