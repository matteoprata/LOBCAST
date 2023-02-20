import numpy as np
from src.data_preprocessing.METACLASS.meta_dataset import MetaDataset
import torch
import src.constants as cst
import os
import json


def load_predictions_fi(num_classes, n_models, n_samples, data_path, k):
    horizons = [1, 10, 2, 3, 5]
    try:
        h_load = horizons.index(int(k))
    except:
        raise Exception("k is not a valid horizon")

    horizons2 = [1, 2, 3, 5, 10]
    h = horizons2.index(int(k))

    entries = os.listdir(data_path)
    all_prob = np.zeros([n_samples, n_models, num_classes])
    all_pred = np.zeros([n_samples, n_models])

    for model in range(n_models):
        file_name = entries[model*len(horizons)+h_load]
        f = open(data_path+file_name)
        data = json.load(f)
        # DEEOLOBATT is saved in another way so we need another load method TODO save DEEEPLOBATT in a normal way
        if 'DEEPLOBATT' in file_name:
            lista = data["LOGITS"].split("]], [[")
            print(len(lista))
            for i in range(len(lista)):
                tmp_pred = lista[i].split("], [")
                pred = []

                for j in range(len(tmp_pred)):
                    tmp = tmp_pred[j].replace("[", "").replace("]", "").split(", ")
                    pred.append(float(tmp[h]))

                np_pred = np.array(pred).astype(np.float)
                all_prob[i, model, :] = np_pred
                all_pred[i, model] = np.argmax(np_pred)

        else:

            lista = data["LOGITS"].split("], [")
            print(len(lista))
            for i in range(len(lista)):
                if i == 0:
                    pred = lista[i][2:].split(", ")
                elif i == len(lista)-1:
                    pred = lista[i][:-2].split(", ")
                else:
                    pred = lista[i].split(", ")

                np_pred = np.array(pred).astype(np.float)
                all_prob[i, model, :] = np_pred
                all_pred[i, model] = np.argmax(np_pred)

    all_prob = all_prob.reshape((n_samples, num_classes*n_models))
    return torch.from_numpy(all_prob).to(cst.DEVICE_TYPE), torch.from_numpy(all_pred).to(cst.DEVICE_TYPE)


def create_datasets_meta_classifier(all_prob_predictions, all_targets, num_classes, n_models):
    print(all_prob_predictions.shape)
    if all_prob_predictions.shape[1] != (num_classes * n_models):
        raise Exception("dimensions of all_prob_predictions are wrong. They have to be [n_models, n_instances]")

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
