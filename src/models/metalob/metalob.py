from datetime import datetime
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import src.constants as cst
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

class MetaLOB(pl.LightningModule):
    def __init__(self, meta_hidden):
        super().__init__()
        input_dim = (len(cst.Models)-1) * cst.NUM_CLASSES

        self.fc1 = nn.Linear(input_dim, meta_hidden)
        self.fc2 = nn.Linear(meta_hidden, cst.NUM_CLASSES)

        self.batch_norm = nn.BatchNorm1d(meta_hidden)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x.shape = [batch, n_models*n_classes]

        o = self.fc1(x)
        # o.shape = [batch, mlp_hidden]

        o = self.batch_norm(self.relu(o))
        # o.shape = [batch, mlp_hidden]

        o = self.fc2(o)
        # o.shape = [batch, n_classes]

        return o


def train_metaLOB(metaLOB, config, lr, momentum, n_samples_train, batch_size, n_epochs, opt, sch, loss_function, data_module):
        best_val_loss = np.inf
        best_test_epoch = 0
        horizon = config.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON]
        print("the horizon is "+ str(horizon))
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        device = cst.DEVICE_TYPE

        for it in tqdm(range(n_epochs)):

            metaLOB.train()
            t0 = datetime.now()
            train_loss = []

            for inputs, targets in train_loader:
                loss = training_step(metaLOB, inputs, targets, opt, loss_function, device)
                train_loss.append(loss)

            # Get mean train loss
            mean_train_loss = np.mean(train_loss)

            metaLOB.eval()
            val_loss = []
            for inputs, targets in val_loader:
                loss = val_step(metaLOB, inputs, targets, loss_function, device)
                val_loss.append(loss)

            mean_val_loss = np.mean(val_loss)

            # We save the best model
            if mean_val_loss < best_val_loss:
                torch.save(metaLOB, f'data/saved_models/metaLOB_k={horizon}.pt')
                best_val_loss = mean_val_loss
                best_test_epoch = it
                print('model saved')

            dt = datetime.now() - t0
            print(f'Epoch {it + 1}/{n_epochs}, Train Loss: {mean_train_loss:.4f}, \
              Validation Loss: {mean_val_loss:.4f}, Duration: {dt}, Best Val Epoch: {best_test_epoch}')

def test_metaLOB(metaLOB, config, lr, momentum, n_samples_train, batch_size, n_epochs, opt, sch, loss, data_module):
    n_correct = 0.
    n_total = 0.
    all_targets = []
    all_predictions = []
    test_loader = data_module.test_dataloader()
    device = cst.DEVICE_TYPE

    for inputs, targets in test_loader:
        targets, predictions = test_step(metaLOB, inputs, targets, device)

        # update counts
        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_predictions = np.concatenate(all_predictions)
    print(classification_report(all_targets, all_predictions, digits=4))
    c = confusion_matrix(all_targets, all_predictions, normalize="true")
    disp = ConfusionMatrixDisplay(c)
    disp.plot()
    plt.show()
    return all_predictions

def training_step(metaLOB, inputs, targets, opt, loss, device):
    # move data to GPU
    inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

    # zero the parameter gradients
    opt.zero_grad()

    # Forward pass
    outputs = metaLOB(inputs)
    loss = loss(outputs, targets)

    # Backward and optimize
    loss.backward()
    opt.step()
    return loss.item()

def val_step(metaLOB, inputs, targets, loss, device):
    inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)
    outputs = metaLOB(inputs)
    loss = loss(outputs, targets)
    return loss.item()

def test_step(metaLOB, inputs, targets, device):
    # Move to GPU
    inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device, dtype=torch.int64)

    # Forward pass
    outputs = metaLOB(inputs)

    # Get prediction
    # torch.max returns both max and argmax
    _, predictions = torch.max(outputs, 1)
    return targets, predictions

def plot_correlation_vector(base_predictions, meta_predictions, horizon):
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(base_predictions.T, meta_predictions)
    corr_vector = np.round(corr_matrix[-1, :-1], decimals=2)
    corr_vector = corr_vector.reshape(1, -1)

    # collect models names
    models = sorted([model.name for model in cst.Models if (model.name != "METALOB")])

    # we swap the order of DeepLOBATT and DeepLOB, because in the json there is DEEPLOBATT first
    models[8], models[9] = models[9], models[8]

    # Create heatmap
    ax = sns.heatmap(corr_vector, vmin=-1, vmax=1, center=0, cmap="coolwarm", annot=True)
    ax.figure.set_size_inches(15, 1)

    # Set axis labels and title
    ax.set_xlabel('Base Classifiers')
    ax.set_xticklabels(models, fontsize=9, rotation=90, ha='center')
    ax.set_ylabel('Meta Classifier')
    ax.set_title(f'Correlation Vector for k={horizon}')

    # Show plot
    plt.show()
    ax.figure.savefig(f"data/correlation_vector_base_meta_K={horizon}.png")





