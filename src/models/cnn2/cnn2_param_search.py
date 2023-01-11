
import src.constants as cst

hyperparameters_cnn2 = {

    cst.LearningHyperParameter.EPOCHS.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.RMSPROP.value]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'max': 0.001, 'min': 0.0001},

    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [16, 32, 64]},  # [32, 64, 128]
    cst.LearningHyperParameter.IS_SHUFFLE.value: {'values': [True]},
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: {'values': [100]},

}
