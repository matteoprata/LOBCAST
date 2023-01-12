
import src.constants as cst

hyperparameters_mlp = {

    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'max': 0.001, 'min': 0.0001},
    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [16, 32, 48]},       # [32, 64, 128]
    cst.LearningHyperParameter.IS_SHUFFLE_TRAIN_SET.value: {'values': [True]},
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: {'values': [100]},
    cst.LearningHyperParameter.MLP_HIDDEN.value: {'values': [128]},  # [128, 192, 256]
    cst.LearningHyperParameter.P_DROPOUT.value: {'values': [0]},  # [0.1, 0.25, 0.5]
}
