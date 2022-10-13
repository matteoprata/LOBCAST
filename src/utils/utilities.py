
import pickle


def read_data(fname):
    with open(fname, 'rb') as handle:
        out_df = pickle.load(handle)
    return out_df


def write_data(data, fname):
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

