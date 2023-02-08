
import src.constants as cst

HP_AXIALLOB = {
    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [50]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.SGD.value]},
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: {'values': [40]},  # [32, 64, 128]
    cst.LearningHyperParameter.MOMENTUM.value: {'values': [0.9]},  # [32, 64, 128]

    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.001, 0.00325, 0.0055, 0.00775, 0.01]},  # {'max': 0.01, 'min': 0.001},  # 'max': 0.001, 'min': 0.0001
    cst.LearningHyperParameter.BATCH_SIZE.value:    {'values': [32, 64, 128]},  # [32, 64, 128]
}

HP_AXIALLOB_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 50,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.SGD.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.009316,
    cst.LearningHyperParameter.BATCH_SIZE.value: 64,
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: 40,
    cst.LearningHyperParameter.MOMENTUM.value: 0.9
}

HP_AXIALLOB_LOBSTER_FIXED = {

}