import argparse
import numpy as np
import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import collections
import pickle

argParser = argparse.ArgumentParser()

argParser.add_argument("--id")
argParser.add_argument("--n_estimators")
argParser.add_argument("--criterion")
argParser.add_argument("--max_depth")
argParser.add_argument("--min_samples_split")
argParser.add_argument("--min_samples_leaf")
argParser.add_argument("--max_features")
argParser.add_argument("--fi_k")

args = argParser.parse_args()

RUN_ID = int(args.id)
n_estimators = int(args.n_estimators)
criterion = args.criterion
max_depth = int(args.max_depth)
min_samples_split = int(args.min_samples_split)
min_samples_leaf = int(args.min_samples_leaf)
max_features = args.max_features
FI_K = int(args.fi_k)

HORIZONS_MAPPINGS_FI = {
    1: -5,
    2: -4,
    3: -3,
    5: -2,
    10: -1
}
NUM_SNAPSHOTS = 100

DATA_FI_FOLDER = 'data/FI-2010/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore'
FILE_PATH_TRAIN = DATA_FI_FOLDER + '/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_7.txt'
FILE_PATHS_TEST = [
    DATA_FI_FOLDER + '/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_7.txt',
    DATA_FI_FOLDER + '/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_8.txt',
    DATA_FI_FOLDER + '/NoAuction_Zscore_Testing/Test_Dst_NoAuction_ZScore_CF_9.txt'
]

dataset = np.loadtxt(FILE_PATH_TRAIN)
n_samples_train = int(np.floor(dataset.shape[1] * 0.8))
dataset_train = dataset[:, :n_samples_train]
dataset_val = dataset[:, n_samples_train:]

dataset_test = np.hstack(
    [np.loadtxt(F_NAME) for F_NAME in FILE_PATHS_TEST]
)

x_train, y_train = dataset_train[:40].T, dataset_train[HORIZONS_MAPPINGS_FI[FI_K]].T
x_val, y_val = dataset_val[:40].T, dataset_val[HORIZONS_MAPPINGS_FI[FI_K]].T
x_test, y_test = dataset_test[:40].T, dataset_test[HORIZONS_MAPPINGS_FI[FI_K]].T
n_samples_train, n_samples_val, n_samples_test = len(y_train), len(y_val), len(y_test)

x_train_snap = np.asarray([
    x_train[i: i+NUM_SNAPSHOTS].flatten()
    for i in tqdm.tqdm(range(n_samples_train-NUM_SNAPSHOTS))
])
y_train_snap = y_train[NUM_SNAPSHOTS:].astype(int)

x_val_snap = np.asarray([
    x_val[i: i+NUM_SNAPSHOTS].flatten()
    for i in tqdm.tqdm(range(n_samples_val-NUM_SNAPSHOTS))
])
y_val_snap = y_val[NUM_SNAPSHOTS:].astype(int)

x_test_snap = np.asarray([
    x_test[i: i+NUM_SNAPSHOTS].flatten()
    for i in tqdm.tqdm(range(n_samples_test-NUM_SNAPSHOTS))
])
y_test_snap = y_test[NUM_SNAPSHOTS:].astype(int)

occurrences = collections.Counter(y_train_snap).items()
class_weight = {
    c: len(y_train_snap) / (3*n_samples_c)
    for c, n_samples_c in occurrences
}

clf = RandomForestClassifier(
    random_state=0,
    n_jobs=-1,
    verbose=1,
    class_weight=class_weight,
    n_estimators=n_estimators,
    criterion=criterion,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    max_features=max_features,
)

clf.fit(x_train_snap, y_train_snap)

y_pred_val = clf.predict(x_val_snap)
y_pred_test = clf.predict(x_test_snap)

performance_dict_val = classification_report(y_true=y_val_snap, y_pred=y_pred_val, output_dict=True)
performance_dict_test = classification_report(y_true=y_test_snap, y_pred=y_pred_test, output_dict=True)

print(classification_report(y_true=y_test_snap, y_pred=y_pred_test))

hyperparameters_dict = dict(
    n_estimators=n_estimators,
    criterion=criterion,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    max_features=max_features,
)

dict_to_save = {
    'hyperparameters_dict': hyperparameters_dict,
    'performance_dict_val': performance_dict_val,
    'performance_dict_test': performance_dict_test,
}

with open(f'data/experiments/random_forest_fi/{RUN_ID}.pkl', 'wb') as f:
    pickle.dump(dict_to_save, f, pickle.HIGHEST_PROTOCOL)
