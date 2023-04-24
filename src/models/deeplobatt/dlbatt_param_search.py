import src.constants as cst

HP_DEEPATT = {
    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    cst.LearningHyperParameter.EPS.value: {'values': [1e-7]},  # 'max': 0.001, 'min': 0.0001
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: {'values': [50]},  # [50, 100]

    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [128]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.0001]}
}

HP_DEEPATT_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.0008267,
    cst.LearningHyperParameter.EPS.value: 1e-07,
    cst.LearningHyperParameter.BATCH_SIZE.value: 128,
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: 50,
}

HP_DEEPATT_LOBSTER_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.0008267,
    cst.LearningHyperParameter.EPS.value: 1e-07,
    cst.LearningHyperParameter.BATCH_SIZE.value: 128,
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: 50,
}
