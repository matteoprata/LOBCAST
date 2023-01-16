
import src.constants as cst

HP_TABL = {

    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [200]},

    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.01]},  # 'max': 0.001, 'min': 0.0001

    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [256]},  # [32, 64, 128]
    cst.LearningHyperParameter.IS_SHUFFLE_TRAIN_SET.value: {'values': [True]},
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: {'values': [10]},

}


HP_TABL_FI_FIXED = {
    # cst.LearningHyperParameter.EPOCHS_UB.value: 200,
    # cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    # cst.LearningHyperParameter.LEARNING_RATE.value: 0.0001,
    # cst.LearningHyperParameter.BATCH_SIZE.value: 32,
    # cst.LearningHyperParameter.MLP_HIDDEN.value: 64,
    # cst.LearningHyperParameter.LSTM_HIDDEN.value: 40,
    # cst.LearningHyperParameter.LSTM_N_HIDDEN.value: 1,
}
