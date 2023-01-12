
import src.constants as cst

hyperparameters_tabl = {

    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [200]},

    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.01]},  # 'max': 0.001, 'min': 0.0001

    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [256]}, # [32, 64, 128]
    cst.LearningHyperParameter.IS_SHUFFLE_TRAIN_SET.value: {'values': [True]},
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: {'values': [10]},

}