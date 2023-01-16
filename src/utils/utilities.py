
import pickle
import os
import json


def read_data(fname):
    with open(fname, 'rb') as handle:
        out_df = pickle.load(handle)
    return out_df


def write_data(data, path, fname):
    with open(path + fname, 'wb') as handle:
        os.makedirs(path, exist_ok=True)
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_json(msg, fname):
    with open(fname, 'w') as fp:
        json.dump(msg, fp)


def read_json(fname):
    data = None
    if os.path.exists(fname):
        with open(fname, 'r') as fp:
            data = json.load(fp)
    return data