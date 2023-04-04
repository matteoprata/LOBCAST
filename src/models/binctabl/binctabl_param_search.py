
import src.constants as cst

HP_BINTABL = {
    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [200]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: {'values': [10]},
    cst.LearningHyperParameter.BATCH_SIZE.value:    {'values': [128]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [1e-3]}
}

# SC = {0.01, 0.005, 0.001, 0.0005, 0.0001}

HP_BINTABL_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 200,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.0005066,
    # cst.LearningHyperParameter.BATCH_SIZE.value: 128,
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: 10,
}

HP_BINTABL_LOBSTER_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 200,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.001,
    cst.LearningHyperParameter.BATCH_SIZE.value: 128,
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: 10,
}
