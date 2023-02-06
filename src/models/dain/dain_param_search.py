
import src.constants as cst

HP_DAIN = {
    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.RMSPROP.value]},
    cst.LearningHyperParameter.IS_SHUFFLE_TRAIN_SET.value: {'values': [True]},
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: {'values': [100]},
    cst.LearningHyperParameter.DAIN_LAYER_MODE.value: {'values': [
        # None,
        # 'avg',
        # 'adaptive_avg',
        # 'adaptive_scale',
        'full'
    ]},
    cst.LearningHyperParameter.MLP_HIDDEN.value: {'values': [512]},
    cst.LearningHyperParameter.P_DROPOUT.value: {'values': [0.5]},

    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [16, 32, 64]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.001, 0.00325, 0.0055, 0.00775, 0.01]},  # {'max': 0.01, 'min': 0.001},  # 'max': 0.001, 'min': 0.0001
}

HP_DAIN_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.RMSPROP.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.0002169,
    cst.LearningHyperParameter.BATCH_SIZE.value: 32,
    cst.LearningHyperParameter.DAIN_LAYER_MODE.value: "full",
    cst.LearningHyperParameter.MLP_HIDDEN.value: 512,
    cst.LearningHyperParameter.P_DROPOUT.value: 0.5
}
