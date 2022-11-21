import torch
import numpy as np
import argparse
import wandb 
import pandas as pd

from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support as score
from torch.utils import data

import torch.nn.functional as F

from src.models.model_executor import MLP
from src.models.cnn1d.cnn1d import CNN1D
from src.models.cnn2d.cnn2d import CNN2D
from src.models.lstm.lstm import LSTM
from src.models.deeplob.deeplob import DeepLob

MODEL_LSTM = "results/LSTM_50_0.001_32_model.pt"
MODEL_CNN = "results/CNN2_50_0.001_32_model.pt"



def load_data(batch_size, horizon, data_dir, type_model, F_IN, L, seed, cod) -> tuple:
    """ Return dataset_train, dataset_val, dataset_test """

    prefix = 'Adv_Test_Dst_NoAuction_DecPre_CF_{}_{}_seed{}_{}_'.format(F_IN, L, seed, cod)
    dec_test1 = np.loadtxt(data_dir + prefix + '7.txt')
    dec_test2 = np.loadtxt(data_dir + prefix + '8.txt')
    dec_test3 = np.loadtxt(data_dir + prefix + '9.txt')
    dec_test = np.hstack((dec_test1, dec_test2, dec_test3))

    dataset_test = Dataset(data=dec_test, k=4, num_classes=3, T=horizon)

    # convert to our format
    for dt in [dataset_test]:
        if type_model == "MLP":
            dt.x = dt.x.reshape(dt.x.shape[0], dt.x.shape[1], -1)
        if type_model == "LSTM":
            dt.x = torch.squeeze(dt.x) 
        dt.y = F.one_hot(dt.y.to(torch.int64), num_classes=3)

    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False)

    return test_loader


def run(wandb_instance, dir_data, dir_results, type_model, n_epochs, horizon, batch_size, learning_rate=1e-3):
    combinations = [(150.0, 5, 'F'), (75.0, 5, 'F'), (37.5, 5, 'F'), 
                    (25.0, 5, 'F'), (18.75, 5, 'F'), (15.0, 5, 'F'),
                    (15, 0, 'L'), (15, 1, 'L'), (15, 2, 'L'), 
                    (15, 3, 'L'), (15, 4, 'L'), (15, 5, 'L')]

    out_dict = {"model" : [], "F" : [], "L" : [], "cod" : [], "seed" : [],
                "Accuracy" : [], "Loss" : [], "Precision" : [], "Recall" : [], "Fscore" : []}
    for type_model in ["CNN2", "LSTM"]:
        for F, L, cod in combinations:
            # for seed in range(1, 21):
                test_loader = load_data(batch_size, horizon, dir_data, type_model, L=L, F_IN=F, seed=seed, cod=cod)
            
                title = type_model + "_" + str(horizon) + " " + str(learning_rate) + "_" + str(batch_size)
                
                n_feat = 40
                n_classes = 3

                test_only = False

                if type_model == 'CNN':
                    temp = 3
                    if horizon == 20:
                        temp = 6
                    elif horizon == 50:
                        temp = 13
                    model = CNN1D(horizon, n_feat, n_classes, temp)
                    model.to(device)
                elif type_model == "CNN2":
                    temp = 1
                    if horizon == 20:
                        temp = 9
                    elif horizon == 50:
                        temp = 99
                    model = CNN2D(horizon, n_feat, n_classes, temp)
                    model.load_state_dict(torch.load(MODEL_CNN))
                    model.eval()
                    model.to(device)
                elif type_model == "DeepLob":
                    model = DeepLob(n_classes)
                    # TODO: it's not reloaded??
                    model.to(device)
                elif type_model == "MLP":
                    model = MLP(n_feat * horizon, n_classes)
                    # TODO: it's not reloaded??
                    model.to(device)
                elif type_model == "LSTM":
                    model = LSTM(n_classes, n_feat, 32, 1, horizon)
                    model.load_state_dict(torch.load(MODEL_LSTM))
                    model.eval()
                    model.to(device)
                else:
                    print("Model not recognized:", type_model)
                    exit(1)

                # specify loss function (categorical cross-entropy)
                criterion = nn.CrossEntropyLoss()

                # specify optimizer (stochastic gradient descent) and learning rate
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

                # initialize tracker for minimum validation loss
                valid_loss_min = np.Inf  # set initial "min" to infinity

                test_loss = []
                test_acc = []
                train_loss = []
                train_acc = []
                best_fscore = -np.inf
                if True:
                
                    batch_loss = 0
                    total_t = 0
                    y_true_v = []
                    y_pred_v = []
                    correct_t = 0
                    with torch.no_grad():
                        model.eval()
                        for data_t, target_t in test_loader:
                            data_t, target_t = data_t.to(device, dtype=torch.float), target_t.to(device,
                                                                                                dtype=torch.float)  # on GPU
                            # if type_model == 'MLP':
                            if type_model not in ["CNN", "DeepLob", "CNN2", "LSTM"]:
                                data_t = data_t.view(data_t.size(0), -1)
                            outputs_t = model(data_t)
                            target_t = torch.argmax(target_t, dim=1)
                            loss_t = criterion(outputs_t, target_t)
                            batch_loss += loss_t.item()
                            pred_t = torch.argmax(outputs_t, dim=1)
                            correct_t += torch.sum(pred_t == target_t).item()
                            total_t += target_t.size(0)
                            y_true_v += [target_t.cpu().data.numpy()]
                            y_pred_v += [pred_t.cpu().data.numpy()]

                        test_accuracy = 100 * (correct_t / total_t)
                        test_acc.append(test_accuracy)
                        test_loss.append(batch_loss / len(test_loader))
                        # network_learned = batch_loss < valid_loss_min
                        print(f'test loss: {np.mean(test_loss):.4f}, test acc: {test_accuracy:.4f}%\n')
                        y_true_v = np.concatenate(y_true_v)
                        y_pred_v = np.concatenate(y_pred_v)
                        precision_v, recall_v, fscore_v, support_v = score(y_true_v, y_pred_v, average='macro')
                        
                        out_dict["model"] += [type_model]
                        out_dict["F"] += [F]
                        out_dict["L"] += [L]
                        out_dict["cod"] += [cod]
                        out_dict["seed"] += [seed]
                        out_dict["Accuracy"] += [test_accuracy]
                        out_dict["Loss"] += [np.mean(test_loss)]
                        out_dict["Precision"] += [precision_v]
                        out_dict["Recall"] += [recall_v]
                        out_dict["Fscore"] += [fscore_v]                        
                    
                        """
                        if network_learned:
                            if wandb_instance is not None:
                                torch.save(model.state_dict(), wandb_instance.dir + title + '_model.pt')
                            valid_loss_min = batch_loss
                            torch.save(model.state_dict(), dir_results + title + '_model.pt')
                            print('Detected network improvement, saving current model')
                        """

                df_out = pd.DataFrame(out_dict)
                print(df_out)
                df_out.to_csv("attack_metrics.csv")



def prepare_x(data):
    df1 = data[:40, :].horizon
    return np.array(df1)


def get_label(data):
    lob = data[-5:, :].horizon
    return lob


def data_classification(X, Y, T):
    [N, D] = X.shape
    df = np.array(X)

    dY = np.array(Y)

    dataY = dY[T - 1:N]

    dataX = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        dataX[i - T] = df[i - T:i, :]

    return dataX, dataY


def torch_data(x, y):
    x = torch.from_numpy(x)
    x = torch.unsqueeze(x, 1)
    y = torch.from_numpy(y)
    y = F.one_hot(y, num_classes=3)
    return x, y


class Dataset(data.Dataset):
    """ Characterizes a dataset for PyTorch """

    def __init__(self, data, k, num_classes, T):
        """Initialization"""
        self.k = k
        self.num_classes = num_classes
        self.T = T

        x = prepare_x(data)
        y = get_label(data)
        x, y = data_classification(x, y, self.T)
        y = y[:, self.k] - 1
        self.length = len(x)

        x = torch.from_numpy(x)
        self.x = torch.unsqueeze(x, 1)
        unique, counts = np.unique(y, return_counts=True)
        print(dict(zip(unique, counts)))
        self.y = torch.from_numpy(y)

    def __len__(self):
        """Denotes the total number of samples"""
        return self.x.shape[0]

    def __getitem__(self, index):
        """Generates samples of data"""
        return self.x[index], self.y[index]


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def normal_exec():
    run(None)

def wand_db_exec():
    # -- WANDB --
    project_name = "adversarial_attacks"
    with wandb.init(project=project_name) as wandb_instance:
        wandb_hyperparams_config = wandb_instance.config
        dir_data = wandb_hyperparams_config["dir_data"]
        dir_results = wandb_hyperparams_config["dir_results"]
        type_model = wandb_hyperparams_config["type_model"]
        n_epochs = wandb_hyperparams_config["n_epochs"]
        horizon = wandb_hyperparams_config["horizon"]
        batch_size = wandb_hyperparams_config["batch_size"]
        learning_rate = wandb_hyperparams_config["learning_rate"]
        # resume = wandb_hyperparams_config["resume"]
        
        run(wandb_instance, dir_data, dir_results, type_model, n_epochs, horizon, batch_size, learning_rate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('dir_results', metavar='DIR', help='path to save results')
    parser.add_argument('dir_data', metavar='DIR', help='path to dataset')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-k', '--horizon', default=50, type=int, metavar='N', help='horizon can be 10, 20 or 50',
                        choices=[10, 20, 50])
    parser.add_argument('-m', '--type_model', default="MLP", type=str,
                        help='Name of the type_model, can be MLP, CNN, CNN2, LSTM or DeepLob (default: MLP)',
                        choices=["MLP", "CNN", "CNN2", "DeepLob", "LSTM"])
    # parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to the best model (default: none)')
    args = parser.parse_args()

    dir_data = args.dir_data
    dir_results = args.dir_results
    type_model = args.type_model
    n_epochs = args.epochs
    horizon = args.horizon
    batch_size = args.batch_size
    # resume = args.resume

    run(None, dir_data, dir_results, type_model, n_epochs, horizon, batch_size)