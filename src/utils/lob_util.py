# NOTICE: on ubuntu/linux please install : "apt-get install libarchive-dev" and then  use "pip3 install libarchive"
import re
import pandas as pd
import numpy as np
import sys
import os

from enum import Enum

import tqdm
from matplotlib import pyplot as plt
from matplotlib import colors
from datetime import datetime, timedelta
from collections import Counter
from plotly import graph_objs as go

import src.config as config

# ---- DATASET CONVERTION ----

# txt to lob

def f1_file_dataset_to_lob_df(dataset_filename : str, label_type : int =4):
    """ The function load the data from dataset_filename
        as np.darray in F1 format (see here: https://arxiv.org/pdf/1705.03233.pdf) 
        and convert them to a simple dataframe with 41 columns (4features x 10 level + label).
        
        
        label_type (int) : the kind of label to use, the dataset has 5 labels.
        
        Notice that, if the data are already normalized, we don't un-normalize them.
    """
    array_data = np.loadtxt(dataset_filename)
    return f1_dataset_to_lob_df(array_data)


def f1_dataset_to_lob_df(array_data : np.ndarray, label_type : int = 4) -> pd.DataFrame:
    """ The function load the data as np.darray in F1 format (see here: https://arxiv.org/pdf/1705.03233.pdf) 
        and convert them to a simple dataframe with 41 columns (4features x 10 level + label).
        
        label_type (int) : the kind of label to use, the dataset has 5 labels.
        
        Notice that, if the data are already normalized, we don't un-normalize them.
    """
    nfeatures = 40
    label_pos = -5
    col_base_names = ["psell", "vsell", "pbuy", "vbuy"]
    col_names  = [c + str(i) for i in range(1, 11) for c in col_base_names]
    y_name = "y"
    
    # extract data correctly
    raw_df_features = array_data[:nfeatures, :].horizon
    raw_lob_label = array_data[label_pos:, :].horizon
    raw_lob_label = raw_lob_label[:, label_type] - 1

    # put them in df format
    df_features = pd.DataFrame(raw_df_features, columns=col_names)
    df_lob_label = pd.DataFrame(raw_lob_label, columns=[y_name])
    df_features[y_name] = df_lob_label[y_name]

    return df_features

# lob to txt 

def lob_df_to_f1_dataset_file(df_lob : pd.DataFrame, filename : str):
    """ The function save the data as np.darray in F1 format but with only 41 rows.
        (see here: https://arxiv.org/pdf/1705.03233.pdf) 
        The other rows are fake data.
        We convert the dataframe to the F1 format 
                
        Notice that, if the data are not normalized, we don't normalize them.
    """
    # convert data 
    df_lob = df_lob.copy()
    base_data = lob_df_to_f1_dataset(df_lob) 
    np.savetxt(filename, base_data)
    
              
def lob_df_to_f1_dataset(df_lob : pd.DataFrame) -> np.ndarray:
    """ The function return the data as np.darray in F1 format but with only 41 rows.
        (see here: https://arxiv.org/pdf/1705.03233.pdf) 
        The other rows are fake data.
        We convert the dataframe to the F1 format 
                
        Notice that, if the data are not normalized, we don't normalize them.
    """
    nfeatures = 40
    extra_fake_features = 108
    label_name = "y"
    
    # add extra dummy data 
    for i in range(extra_fake_features):
        df_lob.insert(nfeatures, 'dummy_col' + str(i), np.nan)
    
    # change the goal format
    df_lob[label_name] = df_lob[label_name] + 1
    
    # extract data correctly
    base_data = df_lob.values
    
    #Transpose data
    base_data = base_data.T
    
    return base_data



#  ---- plot stuff ----- #

def load_column_from_ohcl(symbols, format_file, column="close", pickled_data=False) -> pd.DataFrame:
    """
    Read a column for a bunch of sumbols

    :param symbols: the symbols to read into a unique df
    :param format_file: the format of the files (e.g., old_data/ohcl_data/sec/{}.bz2)
    :param column: the column to read for each symbol (the column will be renamed by name symbol)
    :return: the dataframe
    """
    out_df = None
    for sym in symbols:
        if pickled_data:
            df = pd.read_pickle(format_file.format(sym))
        else:
            df = pd.read_pickle(format_file.format(sym))
        if out_df is None:
            out_df = df
            out_df[sym] = out_df[column]
            out_df = out_df[[sym]]
        else:
            out_df[sym] = df[column]
    return out_df


def save_df_to_bz(df: pd.DataFrame, filename: str) -> None:
    """ save a dataframe with an efficient compression into filename file (please use .bz2 extension)

    :param df: the dataframe to save
    :param filename: the path where save the dataframe (use .bz2 extension)
    :return: nothing
    """
    assert ".bz2" in filename
    df.to_pickle(filename)


def load_df_from_bz(filename: str) -> pd.DataFrame:
    return pd.read_pickle(filename)


def add_roi_spike(df: pd.DataFrame, window: int = 20, column: str = "close"):
    df["spike_lev"] = (df[column].shift(-window) - df[column]) / df[column]
    return df


def plot_symbols(list_df: list, plot: bool = True, filename: str = None) -> None:
    """
    Plot or Save the "close" of each period for the symbols

    :param list_df: the list of symbols to plot/save. The list is: [(df1, synname1), (df2, symname2), ...]
    :param plot: whether plot or not the symbols
    :param filename: wheter save or not (and where) the plot
    :return:  None
    """
    # Create traces
    fig = go.Figure()
    for df, syn_name in list_df:
        fig.add_trace(go.Scatter(x=df.index, y=df["close"] / float(df[1:2]["close"]),
                                 mode='lines+markers',
                                 name=syn_name))
    if filename is not None:
        fig.write_html(filename)
    if plot:
        fig.show()


def plot_candlestick(df: pd.DataFrame, filename: str = None, plot: bool = True) -> None:
    """
    use plotly to plot candle stick

    :param df: the input old_data, the df should contains: date, open, high, low, close
    :param filename: where (if not None) save the output html file
    :param plot: it display or not the old_data
    :return: None
    """
    trace1 = go.Candlestick(x=df.index,
                            open=df['open'],
                            high=df['high'],
                            low=df['low'],
                            close=df['close'], name="Candlestick")

    # add volume
    trace2 = go.Bar(x=df.index, y=df["volume"], name='Volume')

    # add orders
    trace3 = go.Bar(x=df.index, y=df["norders"], name='Number of orders')

    go.Bar()
    fig = make_subplots(rows=5, cols=1,
                        specs=[
                            [{'rowspan': 2}],
                            [None],
                            [{}],
                            [{}],
                            [{}],
                        ])
    fig.update_layout(xaxis_rangeslider_thickness=0.04)
    # Add range slider
    fig.add_trace(trace1)
    fig.add_trace(trace2, row=4, col=1)
    fig.add_trace(trace3, row=5, col=1)
    fig['layout'].update(title="Candlestick Chart", xaxis=dict(tickangle=-90
                                                               ))

    if plot:
        fig.show()
    # if filename is not None:
    fig.write_html("prova.html")


def candlestick_from_7z(sym_7z: str, startdate: str, lastdate: str, granularity: config.Granularity) -> None:
    """ use plotly to plot candle stick

        the input df should contains: date, open, high, low, close
    """
    gd.from_7z_to_unique_df(sym_7z, startdate, lastdate, granularity=granularity, plot=True, level=1)


def candlestick_test() -> None:
    """ use plotly to plot candle stick

        the input df should contains: date, open, high, low, close
    """
    start_date = datetime.strptime("06.02.2018", "%d.%m.%Y")
    df_o = pd.read_csv("old_data/lobster_data/AAPL_2020-10-07_34200000_57600000_orderbook_1.csv")
    df_m = pd.read_csv("old_data/lobster_data/AAPL_2020-10-07_34200000_57600000_message_1.csv")
    gd.lobster_to_ohlc(df_m, df_o, start_date, granularity=config.Granularity.Min5, plot=True)


def plot_all_test() -> None:
    """ test the plot all code """
    start_date = datetime.strptime("06.02.2018", "%d.%m.%Y")
    df_o = pd.read_csv("old_data/lobster_data/AAPL_2020-10-07_34200000_57600000_orderbook_1.csv")
    df_m = pd.read_csv("old_data/lobster_data/AAPL_2020-10-07_34200000_57600000_message_1.csv")
    df_1 = gd.lobster_to_ohlc(df_m, df_o, start_date, granularity=config.Granularity.Min1)

    df_o = pd.read_csv("old_data/lobster_data/TAP_2018-02-06_34200000_57600000_orderbook_1.csv")
    df_m = pd.read_csv("old_data/lobster_data/TAP_2018-02-06_34200000_57600000_message_1.csv")
    df_2 = gd.lobster_to_ohlc(df_m, df_o, start_date, granularity=config.Granularity.Min1)

    df_o = pd.read_csv("old_data/lobster_data/TSLA_2018-09-07_34200000_57600000_orderbook_1.csv")
    df_m = pd.read_csv("old_data/lobster_data/TSLA_2018-09-07_34200000_57600000_message_1.csv")
    df_3 = gd.lobster_to_ohlc(df_m, df_o, start_date, granularity=config.Granularity.Min1)

    plot_symbols([(df_1, "apple"), (df_2, "tap"), (df_3, "tesla")])


def gradient_color(lenght: int, cmap: str = "brg") -> list:
    """
    :param lenght: the len of colors to create
    :return: a list of different matplotlib colors
    """
    t_colors = []
    paired = plt.get_cmap(cmap)
    for i in range(lenght):
        c = paired(i / float(lenght))
        t_colors += [colors.to_hex(c)]
    return t_colors



pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)

COLUMNS_NAMES = {"orderbook": ["sell", "vsell", "buy", "vbuy"],
                 "message": ["time", "event_type", "order_id", "size", "price", "direction", "unk"]}

def message_columns():
    """ return the message columns for the LOBSTER orderbook """
    return ["time", "event_type", "order_id", "size", "price", "direction", "unk"]


def orderbook_columns(level: int):
    """ return the column names for the LOBSTER orderbook, acording the input level """
    orderbook_columns = []
    for i in range(1, level + 1):
        orderbook_columns += ["psell" + str(i), "vsell" + str(i), "pbuy" + str(i), "vbuy" + str(i)]
    return orderbook_columns


def from_folder_to_unique_df(
        file_7z: str,
        first_date: str = "1990-01-01",
        last_date:  str = "2100-01-01",
        plot: bool = False, level: int = 10,
        path: str = "",
        granularity: config.Granularity = config.Granularity.Sec1,
        add_messages=False,
        boundaries_purge=0):
    """ return a unique df with also the label

        add_messages : if True keep messages along the orderbook data. It does not work with granularity != None

    """
    message_dfs = read_sub_routine(file_7z, first_date, last_date, "message", level=level, path=path)
    orderbook_dfs = read_sub_routine(file_7z, first_date, last_date, "orderbook", level=level, path=path)
    frames = []

    assert list(message_dfs.keys()) == list(orderbook_dfs.keys()), "the messages and orderbooks have different days!!"
    print("Iterating over trading days...")
    for d in tqdm.tqdm(message_dfs.keys()):

        tmp_df = lobster_to_gran_df(
            message_dfs[d],
            orderbook_dfs[d],
            d,
            granularity=granularity,
            level=level,
            add_messages=add_messages,
            boundaries_purge=boundaries_purge)

        frames.append(tmp_df)

    # stacks all the days one on top of the other
    result = pd.concat(frames, ignore_index=False)
    return result


def read_sub_routine(file_7z: str, first_date: str = "1990-01-01",
                     last_date: str = "2100-01-01",
                     type_file: str = "orderbook",
                     level: int = 10,
                     path: str = "") -> dict:
    """
        :param file_7z: the input file where the csv with old_data are stored
        :param first_date: the first day to load from the input file
        :param last_date: the last day to load from the input file
        :param type_file: the kind of old_data to read. type_file in ("orderbook", "message")
        :param level: the LOBSTER level of the orderbook
        :param path: data path
        :return: a dictionary with {day : dataframe}
    """
    assert type_file in ("orderbook", "message"), "The input type_file: {} is not valid".format(type_file)

    columns = message_columns() if type_file == "message" else orderbook_columns(level)
    # if both none then we automatically detect the dates from the files
    first_date = datetime.strptime(first_date, "%Y-%m-%d")
    last_date = datetime.strptime(last_date, "%Y-%m-%d")

    all_period = {}  # day :  df

    path = path + file_7z
    print("Reading all", type_file, "files...")
    for file in tqdm.tqdm(sorted(os.listdir(path))):
        # print("processed file", path, file)
        # read only the selected type of file
        if type_file not in str(file):
            continue

        # read only the old_data between first_ and last_ input dates
        m = re.search(r".*([0-9]{4}-[0-9]{2}-[0-9]{2}).*", str(file))
        if m:
            entry_date = datetime.strptime(m.group(1), "%Y-%m-%d")
            if entry_date < first_date or entry_date > last_date:
                continue
        else:
            print("error for file: {}".format(file))
            continue

        curr = path + '/' + file

        # inferring type has a high memory usage low_memory can't be true as default
        df = pd.read_csv(curr, names=columns, low_memory=False)
        all_period[entry_date] = df

    return all_period


def lobster_to_sec_df_from_files(message_file, orderbook_file,
                                 datetime_str="01.10.2020",
                                 granularity: config.Granularity = config.Granularity.Sec1):
    """ create a dataframe with midprices, sell and buy for each second

        message_file : a csv file with the messages (lobster old_data format) without initial start lob
        ordebook_file : a csv file with the orderbook (lobster old_data format) without initial start lob
        datetime_str : should be a start date in the message file and orderbook file, with %d.%m.%Y format
        granularity : the granularity to use in the mid-prices computation
        plot : whether print or not the mid_prices
    """
    start_date = datetime.strptime(datetime_str, "%d.%m.%Y")

    order_df = pd.read_csv(orderbook_file)
    message_df = pd.read_csv(message_file)

    lobster_to_gran_df(message_df, order_df, datetime_start=start_date, granularity=granularity)


def lobster_to_gran_df(
        message_df,
        orderbook_df,
        datetime_start: datetime,
        granularity: config.Granularity = config.Granularity.Sec1,
        level: int = 10,
        add_messages=False,
        boundaries_purge=0
    ):
    """ create a dataframe with midprices, sell and buy for each second

        message_df : a csv df with the messages (lobster old_data format) without initial start lob
        ordebook_df : a csv df with the orderbook (lobster old_data format) without initial start lob
        datetime_start : should be a start date in the message file and orderbook file
        granularity : the granularity to use in the mid-prices computation
        plot : whether print or not the mid_prices
        level : the level of the old_data
        add_messages : if True keep messages along the orderbook data. It does not work with granularity != None
        boundaries_purge : time delta time units to crop when the market starts and before it ends, default 30 mins
    """
    start_date = datetime_start

    # to be sure that columns are okay
    orderbook_df.columns = orderbook_columns(level)
    message_df.columns = message_columns()

    # convert the time to seconds and structure the df to the input granularity
    orderbook_df["seconds"] = message_df["time"]
    orderbook_df["date"] = [start_date + timedelta(seconds=i) for i in orderbook_df["seconds"]]

    if 'Events' in granularity.name or (add_messages and granularity is None):
        orderbook_df[message_df.columns] = message_df[message_df.columns]
        accepted_orders = [o.value for o in (config.OrderEvent.EXECUTION, config.OrderEvent.SUBMISSION, config.OrderEvent.HIDDEN_EXECUTION)]
        orderbook_df = orderbook_df[orderbook_df["event_type"].isin(accepted_orders)]

    if 'Events' in granularity.name:
        orderbook_df = orderbook_df.drop(list(message_df.columns), axis=1)
        orderbook_df = orderbook_df[::granularity.value]
    else:
        if granularity is not None:
            orderbook_df.set_index("date", inplace=True)
            orderbook_df = orderbook_df.resample(granularity.value).first()
            orderbook_df.reset_index(inplace=True)

    assert not boundaries_purge > 0 or granularity == config.Granularity.Sec1, "Unhandled boundaries_purge."

    orderbook_df = orderbook_df.sort_values(by="date").reset_index(drop=True).copy()
    orderbook_df.drop(columns=['seconds'], inplace=True)
    orderbook_df = orderbook_df.set_index('date')

    # removes the first and last *boundaries_purge time units in the dataframe
    purge = pd.Timedelta(boundaries_purge, "sec")
    orderbook_df = orderbook_df[orderbook_df.index.values[0] + purge: orderbook_df.index.values[-1] - purge]
    return orderbook_df



# Threshold dinamica 

# ------------------ **/**   (sarà un trend positivo)
# /\/\/\/\/\/\/\/\/  **/** (sarà un trend stable) 


# Threshold statica

# / è un trend positivo
# \ è un trend negativo 
# ~ and - è un trend stable 

# ------------------ **~**   (sarà un trend ***STABLE***)
# /\/\/\/\/\/\/\/\/  **/** (sarà un trend ***POSITIVE***)


def add_lob_labels_rolling(df : pd.DataFrame, rolling_tu : int, ratio_rolling_window : int = 3600):
    """    
        The new label with dynamic threshold (rooling based on 1 h)

    Args:
        df (pd.DataFrame): [description]
        rolling_tu (int): how big is the window to compute the rolling mean
        ratio_rolling_window (int) : the seconds to compute mean and std of mid-price ratio. Put -1 to disable it.

    Returns:
        [type]: [description]
    """

    # sell = [c for c in df.columns if c.startswith('psell')]
    # buy = [c for c in df.columns if c.startswith('pbuy')]
    df['midprice'] = (df['pbuy1'] + df['psell1']) / 2

    # added for class
    df["m+t"] = df["midprice"].rolling(rolling_tu).mean().shift(-rolling_tu + 1)  # round(df["midprice"].rolling(rolling_tu).mean().shift(-rolling_tu + 1))
    # BE AWARE on GAN! 
    df = df[0:len(df) - (rolling_tu-1)]
    df["ratio_y"] = (df["m+t"] - df["midprice"]) / df["midprice"]  # return

    if ratio_rolling_window == -1:  # this means that we are disabling the rolling ratio 
        ratio_y_rolled_std = df["ratio_y"].std()
        ratio_y_rolled_mean = df["ratio_y"].mean()
        df["y"] = np.where((df["ratio_y"] - ratio_y_rolled_mean) <= -ratio_y_rolled_std, -1, (np.where((df["ratio_y"] - ratio_y_rolled_mean) >= ratio_y_rolled_std, 1, 0)))
    else:
        df["ratio_y_rolled_std"] = df["ratio_y"].rolling(ratio_rolling_window).std().shift(-ratio_rolling_window + 1)
        df["ratio_y_rolled_mean"] = df["ratio_y"].rolling(ratio_rolling_window).mean().shift(-ratio_rolling_window + 1)

        # BE AWARE on GAN! 
        df = df[df["ratio_y_rolled_std"].notna()]  # removes the last 'rolling_tu' nan rows
        df = df[df["ratio_y_rolled_mean"].notna()]  # removes the last 'rolling_tu' nan rows

        df["y"] = np.where((df["ratio_y"] - df["ratio_y_rolled_mean"]) <= -df["ratio_y_rolled_std"], -1,
                        (np.where((df["ratio_y"] - df["ratio_y_rolled_mean"]) >= df["ratio_y_rolled_std"], 1, 0)))

    df["y"] += 1
    
    # pd.set_option('float_format', '{:f}'.format)
    # from collections import Counter
    # df = df.reset_index(drop=True)
    # df["midprice"].plot()
    # plt.savefig("pippo.png")
    # print(Counter(df["y"].values))
    # exit()
    #print(df[["m+t", "midprice", "ratio_y", "ratio_y_rolled_mean", "y"]].to_string())
    #exit()
    
    return df


def add_lob_labels(df: pd.DataFrame, rolling_tu: int, sign_threshold: float):
    df['midprice'] = (df['pbuy1'] + df['psell1']) / 2

    # added for class
    df["m+t"] = df["midprice"].rolling(rolling_tu).mean().shift(- rolling_tu + 1)
    df = df[0:len(df) - (rolling_tu - 1)]

    df["ratio_y"] = (df["m+t"] - df["midprice"]) / df["midprice"]

    df["y"] = np.zeros(df.shape[0])
    df["y"] = np.where(df["ratio_y"] > sign_threshold, 1, df["y"])
    df["y"] = np.where(df["ratio_y"] < -sign_threshold, -1, df["y"])
    df["y"] += 1
    return df
