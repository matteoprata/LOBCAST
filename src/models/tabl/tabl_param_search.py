
import src.constants as cst

HP_TABL = {
    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [200]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: {'values': [10]},

    cst.LearningHyperParameter.BATCH_SIZE.value:    {'values': [256]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [1e-2]}
}

HP_TABL_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 200,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.000867,
    # cst.LearningHyperParameter.BATCH_SIZE.value: 128,
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: 10,
}

HP_TABL_LOBSTER_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 200,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.01,
    cst.LearningHyperParameter.BATCH_SIZE.value: 256,
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: 10,
}