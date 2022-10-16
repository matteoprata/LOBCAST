import torch
import numpy as np
import argparse
import wandb 

from matplotlib import pyplot as plt
from sklearn import metrics
from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support as score
from collections import Counter

import torch.nn.functional as F

from src.models.model_executor import MLP
from src.models.cnn1 import CNN
from src.models.cnn2 import CNN2
from src.models.lstm.lstm import LSTM
from src.models.deeplob import DeepLob
from src.models.cnn_lstm import CNN_LSTM
from src.utils.lobdataset import LOBDataset, DEEPDataset


def load_data(batch_size, horizon, base_lob_dts, type_model, polluded) -> tuple:
    """ Return dataset_train, dataset_val, dataset_test """
    # Convert to Dataset instance
    dataset_train = base_lob_dts.split_train_data(torch_dataset=True)
    dataset_val = base_lob_dts.split_val_data(torch_dataset=True)
    dataset_test = base_lob_dts.split_test_data(torch_dataset=True) 


    ## Try to improve dataset balance using WeightedRandomSampler. We randomly take instances
    # according the class frequency, so that our training set is more balanced.  
    bal_sampler = False # false if training with fi-2010
    oversampling = False
    undersampling = False
    undersampling_w_replacement = False

    if bal_sampler:
        class_sample_count = Counter([int(x) for x in dataset_train.y])
        class_sample_count = [class_sample_count[c] for c in range(3)]
        weights = 1 / torch.Tensor(class_sample_count)
        samples_weight = np.array([weights[int(x)] for x in dataset_train.y])
        samples_weight = torch.from_numpy(samples_weight)
        # oversampling = False
        # undersampling = True
        # undersampling_w_replacement = False
        if oversampling:
            # samples to pick == len(training_set): we do OVERSAMPLING
            samples_to_pick = len(samples_weight)   
            sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, samples_to_pick, replacement=True)
        elif undersampling:
            torch_perm = torch.randperm(dataset_train.y.size()[0])
            dataset_train.x = dataset_train.x[torch_perm] 
            dataset_train.y = dataset_train.y[torch_perm] 
            # samples_to_pick == min(class_sample_count) * 3 : we do UNDERSAMPLING
            # dataset = { label_-1  : 130, label_0 : 5000, label_1 : 200}
            # dataset_completo = 5330
            # sampler take ranomly from dataset_completo
            # le probabilità per ogni elemento in dataset_completo (i.e., 5330 probabilità)
            # il numero di elementi da prendere nel nuovo dataset. 
            min_class_count = min(class_sample_count) 
            # FIX PROBABILITIES FOR EACH CLASS
            counters = {c : 0 for c in range(3)}
            samples_weight = []
            for x in dataset_train.y:
                label_x = int(x)
                counters[label_x] += 1
                if counters[label_x] <= min_class_count:
                    samples_weight += [1]
                else:
                    samples_weight += [0]
            
            samples_weight = torch.from_numpy(np.array(samples_weight))
            samples_to_pick = min_class_count * 3  # (5 instead of 3 to improve the quantity of data    
            sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, samples_to_pick, replacement=False)
        elif undersampling_w_replacement:
            # samples_to_pick == min(class_sample_count) * 3 : we do UNDERSAMPLING
            samples_to_pick = min(class_sample_count) * 5  # (5 instead of 3 to improve the quantity of data
            sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, samples_to_pick, replacement=True)
        ## End sampler  --------------
    
    # Test over/under sampling:
    test_sampling = False

    # convert to our format
    for dt in [dataset_train, dataset_val, dataset_test]:
        if type_model == "MLP":
            dt.x = dt.x.reshape(dt.x.shape[0], dt.x.shape[1], -1)
        if type_model == "LSTM":
            dt.x = torch.squeeze(dt.x) 
        if not test_sampling:
            dt.y = F.one_hot(dt.y.to(torch.int64), num_classes=3)

    # print(dataset_train.x.shape, dataset_train.y.shape)
    
    # Create dataloader
    if oversampling or undersampling or undersampling_w_replacement:
        train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, drop_last=True, sampler=sampler)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size, drop_last=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)
    
    if test_sampling:
        count = {0 : 0, 1 : 0, 2 : 0}
        for batch_idx, (data_, target_) in enumerate(train_loader):
            for t in target_:
                count[int(t)] += 1
        
        print(count)
        exit()

    return train_loader, val_loader, test_loader


def run(wandb_instance, dir_data, base_lob_dts, dir_results, type_model, n_epochs, horizon, batch_size, learning_rate):
    train_loader, val_loader, test_loader = load_data(batch_size, horizon, base_lob_dts, type_model, False)
    
    title = type_model + "_" + str(horizon) + "_" + str(learning_rate) + "_" + str(batch_size)
    
    n_feat = 40
    n_classes = 3

    if type_model == 'CNN':
        temp = 26 # horizon = 100
        if horizon == 10:
            temp = 3
        if horizon == 20:
            temp = 6
        elif horizon == 50:
            temp = 13
        model = CNN(horizon, n_feat, n_classes, temp)
        model.to(device)
    elif type_model == "CNN2":
        temp = 249 # 271
        if horizon == 10:
            temp = 1
        if horizon == 20:
            temp = 9 # 31
        elif horizon == 50:
            temp = 121
        model = CNN2(horizon, n_feat, n_classes, temp)
        model.to(device)
    elif type_model == "DeepLob":
        model = DeepLob(n_classes)
        model.to(device)
    elif type_model == "MLP":
        model = MLP(n_feat * horizon, n_classes)
        model.to(device)
    elif type_model == "LSTM":
        model = LSTM(n_classes, n_feat, 32, 1)
        model.to(device)
    elif type_model == "CNN_LSTM":
        model = CNN_LSTM(horizon, n_feat, n_classes, 64, 1)
        model.to(device)
    else:
        print("Model not recognized:", type_model)
        exit(1)

    # specify loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()

    # specify optimizer (stochastic gradient descent) and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loss = []
    train_acc = []
    total_step = len(train_loader)

    v_loss_list = []
    v_acc_list = []
    
    best_model_name = None
    best_fscore_v = -np.inf
    count_fscore = -1
    file = open(dir_results + 'log_' + title + '.txt', 'w')

    for epoch in range(1, n_epochs + 1):
        running_loss = 0.0
        # scheduler.step(epoch)
        correct = 0
        total = 0
        y_true_t = []
        y_pred_t = []

        y_true_v = []
        y_pred_v = []

        print(f'Epoch {epoch}\n')
        for batch_idx, (data_, target_) in enumerate(train_loader):
            data_, target_ = data_.to(device, dtype=torch.float), target_.to(device, dtype=torch.float)  # on GPU
            # if type_model == 'MLP':
            if type_model not in ["CNN", "DeepLob", "CNN2", "CNN_LSTM", "LSTM"]:
                data_ = data_.view(data_.size(0), -1)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(data_)
            target_ = torch.argmax(target_, dim=1)
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            outputs = torch.argmax(outputs, dim=1)
            correct += torch.sum(outputs == target_).item()
            total += target_.size(0)
            y_true_t += [target_.cpu().data.numpy()]
            y_pred_t += [outputs.cpu().data.numpy()]
            if batch_idx % 20 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch, n_epochs, batch_idx, total_step, loss.item()))
        accuracy = 100 * (correct / total)
        train_acc.append(accuracy)
        train_loss.append(running_loss / total_step)
        print(f'\ntrain loss: {np.mean(train_loss):.4f}, train acc: {accuracy:.4f}%')
        y_true_t = np.concatenate(y_true_t)
        y_pred_t = np.concatenate(y_pred_t)
        precision, recall, fscore, support = score(y_true_t, y_pred_t, average='macro')
        file.writelines([
            f"[TRAIN] Epoch: {epoch}, Accuracy: {accuracy}, Loss: {np.mean(train_loss)}, Precision: {precision}, Recall: {recall}, Fscore: {fscore} \n"])

        # Add metrics to wandb-instance 
        if wandb_instance is not None:
            metrics_dict = { "Train_Accuracy" : accuracy, 
                        "Train_Loss" : np.mean(train_loss),
                        "Train_Precision" : precision, 
                        "Train_Recall" : recall, 
                        "Train_Fscore" : fscore
            }    
            wandb_instance.log(metrics_dict)

        # validation step
        precision_v, recall_v, fscore_v, support_v, accuracy_v, v_acc_list, v_loss_list, y_true_v, y_pred_v = test_model_performance(val_loader, model, True, v_acc_list, v_loss_list, y_true_v, y_pred_v, type_model)

        file.writelines([
            f"[VALIDATION] Epoch: {epoch}, Accuracy: {accuracy_v}, Loss: {np.mean(v_loss_list)}, Precision: {precision_v}, Recall: {recall_v}, Fscore: {fscore_v} \n"])
        
        # Add metrics to wandb-instance 
        if wandb_instance is not None:
            metrics_dict = { "Val_Accuracy" : accuracy_v, 
                    "Val_Loss" : np.mean(v_loss_list),
                    "Val_Precision" : precision_v, 
                    "Val_Recall" : recall_v, 
                    "Val_Fscore" : fscore_v
            }
            wandb_instance.log(metrics_dict)

        # Print the confusion matrix
        file.writelines([str(metrics.confusion_matrix(y_true_v, y_pred_v))])
        # Print the precision and recall, among other metrics
        file.writelines([str(metrics.classification_report(y_true_v, y_pred_v, digits=3))])

        # update best fscore and save model and early stopping
        if fscore_v > best_fscore_v:
            best_fscore_v = fscore_v
            count_fscore = 0
            if wandb_instance is not None:
                print('Detected network improvement, saving best model')
                best_model_name = wandb_instance.dir + title + '_best_model.pt'
                torch.save(model.state_dict(), best_model_name)
                # save locally
                best_model_name = dir_results + title + '_best_model.pt'
                torch.save(model.state_dict(), best_model_name)
            else:
                print('Detected network improvement, saving best model')
                best_model_name = dir_results + title + '_best_model.pt'
                torch.save(model.state_dict(), best_model_name)
            
        elif epoch > 29:
            count_fscore += 1

        last_model_name = dir_results + title + '_model.pt'
        torch.save(model.state_dict(), last_model_name)

        # stop the execution
        if count_fscore == 20:
            model.load_state_dict((torch.load(best_model_name)))
            precision_t, recall_t, fscore_t, support_t, accuracy_t, t_acc_list, t_loss_list, y_true_t, y_pred_t = test_model_performance(test_loader, model, False, [], [], [], [], type_model)  

            file.writelines([
            f"[TEST - EARLY STOP] Epoch: {epoch}, Accuracy: {accuracy_t}, Loss: {np.mean(t_loss_list)}, Precision: {precision_t}, Recall: {recall_t}, Fscore: {fscore_t} \n"])

            plt.figure(figsize=(20, 10))
            plt.title("Train - Val Accuracy")
            plt.plot(train_acc, label='train acc')
            plt.plot(v_acc_list, label='val acc')
            plt.xlabel('num_epochs', fontsize=12)
            plt.ylabel('accuracy', fontsize=12)
            plt.legend()
            plt.savefig(dir_results + title + '_trainval_acc.png')

            plt.figure(figsize=(20, 10))
            plt.title("Train - Val Loss")
            plt.plot(train_loss, label='train loss')
            plt.plot(v_loss_list, label='val loss')
            plt.xlabel('num_epochs', fontsize=12)
            plt.ylabel('loss', fontsize=12)
            plt.legend()
            plt.savefig(dir_results + title + '_trainval_loss.png')

            plt.figure(figsize=(20, 10))
            plt.title("Train - Test Accuracy")
            plt.plot(train_acc, label='train acc')
            plt.plot(t_acc_list, label='test acc')
            plt.xlabel('num_epochs', fontsize=12)
            plt.ylabel('accuracy', fontsize=12)
            plt.legend()
            plt.savefig(dir_results + title + '_traitest_acc.png')

            plt.figure(figsize=(20, 10))
            plt.title("Train - Test Loss")
            plt.plot(train_loss, label='train loss')
            plt.plot(t_loss_list, label='test loss')
            plt.xlabel('num_epochs', fontsize=12)
            plt.ylabel('loss', fontsize=12)
            plt.legend()
            plt.savefig(dir_results + title + '_traitest_loss.png')

            file.close()
            file = open(dir_results + 'log_' + title + '.txt', 'r')
            print(file.read())
            file.close()

            return 

        model.train()

    # test step
    model.load_state_dict((torch.load(best_model_name)))
    precision_t, recall_t, fscore_t, support_t, accuracy_t, t_acc_list, t_loss_list, y_true_t, y_pred_t =  test_model_performance(test_loader, model, False, [], [], [], [], type_model)  

    file.writelines([
            f"[TEST] Epoch: {epoch}, Accuracy: {accuracy_t}, Loss: {np.mean(t_loss_list)}, Precision: {precision_t}, Recall: {recall_t}, Fscore: {fscore_t} \n"])

    plt.figure(figsize=(20, 10))
    plt.title("Train - Val Accuracy")
    plt.plot(train_acc, label='train acc')
    plt.plot(v_acc_list, label='val acc')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.legend()
    plt.savefig(dir_results + title + '_trainval_acc.png')

    plt.figure(figsize=(20, 10))
    plt.title("Train - Val Loss")
    plt.plot(train_loss, label='train loss')
    plt.plot(v_loss_list, label='val loss')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend()
    plt.savefig(dir_results + title + '_trainval_loss.png')

    plt.figure(figsize=(20, 10))
    plt.title("Train - Test Accuracy")
    plt.plot(train_acc, label='train acc')
    plt.plot(t_acc_list, label='test acc')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('accuracy', fontsize=12)
    plt.legend()
    plt.savefig(dir_results + title + '_traintest_acc.png')

    plt.figure(figsize=(20, 10))
    plt.title("Train - Test Loss")
    plt.plot(train_loss, label='train loss')
    plt.plot(t_loss_list, label='test loss')
    plt.xlabel('num_epochs', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.legend()
    plt.savefig(dir_results + title + '_traintest_loss.png')

    file.close()
    file = open(dir_results + 'log_' + title + '.txt', 'r')
    print(file.read())
    file.close()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def test_model_performance(loader, model_object, is_val, acc_list, loss_list, y_true, y_pred, type_model):
    # used for both test and validation
    type_testing = 'val' if is_val else 'test'
    batch_loss = 0
    total_t = 0
    correct_t = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        model_object.eval()
        for data_t, target_t in loader:
            data_t, target_t = data_t.to(device, dtype=torch.float), target_t.to(device,
                                                                                dtype=torch.float)  # on GPU
            if type_model not in ["CNN", "DeepLob", "CNN2", "CNN_LSTM","LSTM"]:
                data_t = data_t.view(data_t.size(0), -1)
            outputs_t = model_object(data_t)
            target_t = torch.argmax(target_t, dim=1)
            loss_t = criterion(outputs_t, target_t)
            batch_loss += loss_t.item()
            
            pred_t = torch.argmax(outputs_t, dim=1)
            correct_t += torch.sum(pred_t == target_t).item()
            total_t += target_t.size(0)
            y_true += [target_t.cpu().data.numpy()]
            y_pred += [pred_t.cpu().data.numpy()]

        accuracy = 100 * (correct_t / total_t)
        acc_list.append(accuracy)
        loss_list.append(batch_loss / len(loader))

        print(f'{type_testing} loss: {np.mean(loss_list):.4f}, {type_testing} acc: {accuracy:.4f}%\n')
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        precision, recall, fscore, support = score(y_true, y_pred, average='macro')

        return precision, recall, fscore, support, accuracy, acc_list, loss_list, y_true, y_pred


def normal_exec():
    run(None)

def wand_db_exec(base_lob_dts, dir_data, dir_results, type_model, n_epochs, horizon, batch_size, learning_rate):
    # -- WANDB --
    project_name = "adversarial_attacks"
    with wandb.init(project=project_name) as wandb_instance:
        
        """
        wandb_hyperparams_config = wandb_instance.config

        dir_data = wandb_hyperparams_config["dir_data"]
        dir_results = wandb_hyperparams_config["dir_results"]
        type_model = wandb_hyperparams_config["type_model"]
        n_epochs = wandb_hyperparams_config["n_epochs"]
        horizon = wandb_hyperparams_config["horizon"]
        batch_size = wandb_hyperparams_config["batch_size"]
        """
        
        wandb_hyperparams_config = {
            "dir_data": dir_data, 
            "dir_results": dir_results, 
            "type_model": type_model, 
            "n_epochs": n_epochs, 
            "horizon": horizon, 
            "batch_size": batch_size, 
            "learning_rate": learning_rate, 
        }
        
        # base_lob_dts = LOBDataset(dir_data, horizon=horizon, sign_threshold=0.002)

        run(wandb_instance, wandb_hyperparams_config["dir_data"], base_lob_dts, 
                wandb_hyperparams_config["dir_results"], wandb_hyperparams_config["type_model"], 
                wandb_hyperparams_config["n_epochs"], wandb_hyperparams_config["horizon"], 
                wandb_hyperparams_config["batch_size"], wandb_hyperparams_config["learning_rate"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('dir_results', metavar='DIR', help='path to save results')
    parser.add_argument('dir_data', metavar='DIR', help='path to dataset')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('-k', '--horizon', default=100, type=int, metavar='N', help='horizon can be 10, 20, 50 or 100',
                        choices=[10, 20, 50, 100])
    parser.add_argument('-m', '--type_model', default="LSTM", type=str,
                        help='Name of the type_model, can be MLP, CNN, CNN2, LSTM, DeepLob or CNN_LSTM',
                        choices=["MLP", "CNN", "CNN2", "DeepLob", "LSTM", "CNN_LSTM"])
    parser.add_argument('-l', '--learning_rate', default=1e-3, type=float, help='The learning rate of the saved_models')
                        
    args = parser.parse_args()

    dir_data = args.dir_data
    dir_results = args.dir_results
    type_model = args.type_model
    n_epochs = args.epochs
    horizon = args.horizon
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    # resume = args.resume
    
    lobster_training = False
    # training with lob data 
    if lobster_training:
        base_lob_dts = LOBDataset(dir_data, horizon=horizon, sign_threshold=0.002)
        # base_lob_dts = LOBDataset(dir_data, horizon=horizon, sign_threshold=0.0005, ratio_rolling_window=-1)
    # training with deep lob data t
    else:
        base_lob_dts = DEEPDataset(dir_data, horizon=horizon)

    wand_db_exec(base_lob_dts, dir_data, dir_results, type_model, n_epochs, horizon, batch_size, learning_rate)
