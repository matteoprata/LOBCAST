
import src.constants as cst

hyperparameters_tlb = {

    cst.LearningHyperParameter.EPOCHS_UB.value: {'values': [100]},

    cst.LearningHyperParameter.OPTIMIZER.value: {'values': [cst.Optimizers.ADAM.value]},
    cst.LearningHyperParameter.LEARNING_RATE.value: {'values': [0.0001]},  # 'max': 0.001, 'min': 0.0001
    cst.LearningHyperParameter.WEIGHT_DECAY.value: {'values': [1e-5]},

    cst.LearningHyperParameter.BATCH_SIZE.value: {'values': [32]},  # [32, 64, 128]
    cst.LearningHyperParameter.IS_SHUFFLE_TRAIN_SET.value: {'values': [True]},
    cst.LearningHyperParameter.NUM_SNAPSHOTS.value: {'values': [100]},

}

