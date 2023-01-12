
import src.constants as cst

hyperparameters_dain = {

    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.RMSPROP.value]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.0001]},

    cst.LearningHyperParameter.IS_SHUFFLE_TRAIN_SET.value: {'values': [True]},
    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [16, 32, 64]},
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: {'values': [100]},

    cst.LearningHyperParameter.DAIN_LAYER_MODE.value: {'values': [
        # None,
        # 'avg',
        # 'adaptive_avg',
        # 'adaptive_scale',
        'full'
    ]},

    cst.LearningHyperParameter.MLP_HIDDEN.value: {'values': [512]},
    cst.LearningHyperParameter.P_DROPOUT.value: {'values': [0.5]}

}
