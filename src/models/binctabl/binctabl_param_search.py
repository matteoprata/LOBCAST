
import src.constants as cst

HP_BINTABL = {
    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [200]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: {'values': [10]},

    cst.LearningHyperParameter.BATCH_SIZE.value:    {'values': [64, 128, 256]},  # [32, 64, 128]
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.0001, 0.002575, 0.00505, 0.007525, 0.01]},  # 'max': 0.001, 'min': 0.0001
}

# SC = {0.01, 0.005, 0.001, 0.0005, 0.0001}

HP_BINTABL_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 200,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.0005066,
    # cst.LearningHyperParameter.BATCH_SIZE.value: 128,
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: 10,
}
