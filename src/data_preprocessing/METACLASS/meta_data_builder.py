from src.data_preprocessing.METACLASS.meta_dataset import MetaDataset
import torch
import src.constants as cst


def load_predictions(num_classes, n_models):
    pred_path = ""
    pred = torch.load(pred_path + "0")
    all_pred = torch.zeros([pred.shape[0], num_classes*n_models], dtype=pred.dtype, device=cst.DEVICE_TYPE)
    all_pred[:, 0] = pred

    for model in n_models-1:
        pred = torch.load(pred_path+str(model+1))
        all_pred[:, model+1] = pred

    return all_pred


def create_datasets_meta_classifier(all_prob_predictions, all_targets, num_classes, n_models):
    if all_prob_predictions.shape[1] != (num_classes * n_models):
        raise Exception("dimensions of all_predictions are wrong. They have to be [n_models, n_instances]")

    dec_size = all_prob_predictions.shape[0]
    train_size, val_size, test_size = int(0.70 * dec_size), int(0.15 * dec_size), int(0.15 * dec_size)

    dec_train = all_prob_predictions[:train_size]
    dec_val = all_prob_predictions[train_size:train_size + val_size]
    dec_test = all_prob_predictions[train_size + val_size:]

    y_train = all_targets[:train_size]
    y_val = all_targets[train_size:train_size + val_size]
    y_test = all_targets[train_size + val_size:]

    meta_train_dataset = MetaDataset(dec_train, y_train, num_classes=3)
    meta_val_dataset = MetaDataset(dec_val, y_val, num_classes=3)
    meta_test_dataset = MetaDataset(dec_test, y_test, num_classes=3)

    return meta_train_dataset, meta_val_dataset, meta_test_dataset
