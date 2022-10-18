
import pickle
import os

def read_data(fname):
    with open(fname, 'rb') as handle:
        out_df = pickle.load(handle)
    return out_df


def write_data(data, path, fname):
    with open(path + fname, 'wb') as handle:
        os.makedirs(path, exist_ok=True)
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

