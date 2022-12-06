
import src.config as co

hyperparameters_tlb = {

    # co.TuningVars.EPOCHS.value: {'values': [5, 10, 15]},

    co.TuningVars.OPTIMIZER.value: {'values': [co.Optimizers.ADAM.value]},
    co.TuningVars.LEARNING_RATE.value: {'values': [0.0001]},  # 'max': 0.001, 'min': 0.0001
    co.TuningVars.WEIGHT_DECAY.value: {'values': [1e-5]},

    co.TuningVars.BATCH_SIZE.value: {'values': [32]},  # [32, 64, 128]
    co.TuningVars.IS_SHUFFLE.value: {'values': [True]},

}

