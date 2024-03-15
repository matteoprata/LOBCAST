
import pickle
import os
import json
import platform, socket, re, uuid, psutil, logging
import matplotlib.pyplot as plt

import inspect


def get_class_arguments(obj):
    signature = inspect.signature(obj)
    parameters = signature.parameters
    arguments = [param.name for param in parameters.values()]  # list of arguments
    return arguments

import src.constants as cst


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
    else:
        print("File", fname, "does not exist.")
    return data


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def get_sys_info():
    info = dict()
    info['platform'] = platform.system()
    info['platform-release'] = platform.release()
    info['platform-version'] = platform.version()
    info['architecture'] = platform.machine()
    info['hostname'] = socket.gethostname()
    info['ip-address'] = socket.gethostbyname(socket.gethostname())
    info['mac-address'] = ':'.join(re.findall('..', '%012x' % uuid.getnode()))
    info['processor'] = platform.processor()
    info['ram'] = str(round(psutil.virtual_memory().total / (1024.0 **3)))+" GB"
    print(info)


def get_sys_mac():
    return ':'.join(re.findall('..', '%012x' % uuid.getnode()))


def get_index_from_window(config):
    if config.DATASET_NAME == cst.DatasetFamily.FI:
        return cst.HORIZONS_MAPPINGS_FI[config.HYPER_PARAMETERS[cst.LearningHyperParameter.FI_HORIZON]]
    elif config.DATASET_NAME == cst.DatasetFamily.LOB:
        return cst.HORIZONS_MAPPINGS_LOBSTER[config.HYPER_PARAMETERS[cst.LearningHyperParameter.FORWARD_WINDOW.value]]


def sample_color(index, cmap='tab10'):
    # 1. Choose your desired colormap
    cmap = plt.get_cmap(cmap)

    # 2. Segmenting the whole range (from 0 to 1) of the color map into multiple segments
    colors = [cmap(x) for x in range(cmap.N)]
    assert index < cmap.N

    # 3. Color the i-th line with the i-th color, i.e. slicedCM[i]
    color = colors[index]
    return color


def sample_marker(index):
    MARKERS = ["s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ".", ",", "o", "v", "^", "<", ">", "1", "2", "3", "4", "8"]
    return MARKERS[index]


def sample_pattern(index):
    MARKERS = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'] + ['/o', '\\|', '|*', '-\\', '+o', 'x*', 'o-', 'O|', 'O.', '*-']
    return MARKERS[index]


def sample_line(index):
    MARKERS = ['-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted', 'loosely dotted', 'densely dotted', 'loosely dashed', 'densely dashed', 'loosely dashdotted', 'densely dashdotted', 'loosely dashdotdotted', 'dashdotdotted', 'densely dashdotdotted']
    return MARKERS[index]


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == '__main__':
    print('mac:', get_sys_mac())
