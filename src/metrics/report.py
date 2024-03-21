import matplotlib.pyplot as plt
import src.constants as cst
from src.utils.utils_generic import read_json


saved_metrics = [
    cst.Metrics.F1.value,
    cst.Metrics.F1_W.value,
    cst.Metrics.PRECISION.value,
    cst.Metrics.PRECISION_W.value,
    cst.Metrics.RECALL.value,
    cst.Metrics.RECALL_W.value,
    cst.Metrics.ACCURACY.value,
    cst.Metrics.MCC.value,
    cst.Metrics.COK.value,
    cst.Metrics.LOSS.value,
]


def plot_metric_training(json_data_path, metric, pdf):
    json_data = read_json(json_data_path)

    # Extract data
    data_train = json_data[cst.ModelSteps.TRAINING.value]
    epochs_train = sorted(map(int, data_train.keys()))
    metric_values_train = [data_train[str(epoch)][metric] for epoch in epochs_train]

    data_val = json_data[cst.ModelSteps.VALIDATION.value]
    epochs_val = sorted(map(int, data_val.keys()))
    metric_values_val = [data_val[str(epoch)][metric] for epoch in epochs_val]

    # Plotting
    plt.figure(figsize=(5, 5))
    plt.plot(epochs_train, metric_values_train, label=cst.ModelSteps.TRAINING.value, marker='.')
    plt.plot(epochs_val, metric_values_val, label=cst.ModelSteps.VALIDATION.value, marker='.')

    plt.title(f'{metric.capitalize()} vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True, alpha=0.2)

    if metric not in [cst.Metrics.LOSS.value, cst.Metrics.CM.value]:
        plt.ylim(-0.05, 1.05)

    plt.tight_layout()
    pdf.savefig(plt.gcf())


def plot_metric_best(json_data_path, metric, pdf):
    json_data = read_json(json_data_path)

    # Extract data
    data_test = json_data[cst.ModelSteps.TESTING.value]
    epochs_test = sorted(map(int, data_test.keys()))
    metric_values_test = [data_test[str(epoch)][metric] for epoch in epochs_test]

    data_val = json_data["validation"]
    epochs_val = sorted(map(int, data_val.keys()))
    metric_values_val = [data_val[str(epoch)][metric] for epoch in epochs_val]

    # Plotting
    plt.figure(figsize=(5, 5))
    plt.bar([cst.ModelSteps.TESTING.value, cst.ModelSteps.VALIDATION.value], metric_values_test + metric_values_val, color=['blue', 'green'])

    plt.title(f'{metric.capitalize()} vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.grid(True, alpha=0.2)

    if metric not in [cst.Metrics.LOSS.value, cst.Metrics.CM.value]:
        plt.ylim(-0.05, 1.05)

    plt.tight_layout()
    pdf.savefig(plt.gcf())
