import src.constants as cst

HP_META_FI_FIXED = {
    cst.LearningHyperParameter.EPOCHS_UB.value: 500,
    cst.LearningHyperParameter.OPTIMIZER.value: cst.Optimizers.SGD.value,
    cst.LearningHyperParameter.LEARNING_RATE.value: 0.01,
    cst.LearningHyperParameter.BATCH_SIZE.value: 32,
    cst.LearningHyperParameter.META_DIM_LAYER: 16
}