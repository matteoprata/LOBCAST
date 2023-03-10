
import src.constants as cst

HP_DEEP = {
    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},
    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    # self.HYPER_PARAMETERS[LearningHyperParameter.EPS] = 1e-08
    cst.LearningHyperParameter.EPS.value: {'values': [1]},
    cst.LearningHyperParameter.BATCH_SIZE.value:    {'values': [32]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.01]}  # {'max': 1e-2, 'min': 1e-5}
}

HP_DEEP_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.0008458,
    cst.LearningHyperParameter.BATCH_SIZE.value: 64,
}

HP_DEEP_LOBSTER_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 100,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.ADAM.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.0001,
    # cst.LearningHyperParameter.BATCH_SIZE.value: 64,
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: 100,
    cst.LearningHyperParameter.BACKWARD_WINDOW.value: 100,
    cst.LearningHyperParameter.FORWARD_WINDOW.value: 100,
}
