
import src.constants as cst

HP_AXIALLOB = {
    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [20]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.SGD.value]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'max': 0.001, 'min': 0.0001},  # 'max': 0.001, 'min': 0.0001
    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [32, 64, 128]},  # [32, 64, 128]
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: {'values': [40]},  # [32, 64, 128]
}

# DONE FIXED 21-01-2023
HP_AXIALLOB_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 20,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.SGD.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.001,
    cst.LearningHyperParameter.BATCH_SIZE.value: 64,
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: 40,
}

HP_AXIALLOB_LOBSTER_FIXED = {
}
