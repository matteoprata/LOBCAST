
import pandas as pd
import numpy as np
from enum import Enum
from matplotlib import pyplot as plt


def is_outlier(X, Q1, Q3, outlier_factor=3):
    """ Says whether the outlier is an extreme. """
    # stack.quantile([0.25, 0.75])
    IQ = Q3 - Q1
    return not Q1 - outlier_factor * IQ < X < Q3 + outlier_factor * IQ


def stationary_normalize_data(data, normalization_mean=None, normalization_std=None):
    """ TODO: remember to use the mean/std of the training set, to z-normalize the test set. """
    col_choice = {"volumes": get_volume_column_name(data.columns), "prices": get_price_column_name(data.columns)}

    print("Normalization...")

    for col_name in col_choice:
        cols = col_choice[col_name]

        mean_out = normalization_mean if normalization_mean is not None else data.loc[:, cols].stack().mean()
        std_out = normalization_std if normalization_std is not None else data.loc[:, cols].stack().std()

        data.loc[:, cols] = (data.loc[:, cols] - mean_out) / std_out
        data.loc[:, cols] = data.loc[:, cols]

        # TODO: volumes and prices can be negative, add min value
        # + abs(data.loc[:, cols].stack().min())  # scale positive
        # data.loc[:, cols].stack().plot.hist(bins=200, alpha=0.5, title=col_name)
        # plt.show()

    data = data.fillna(method="bfill")
    data = data.fillna(method="ffill")
    return data


class DataCols(Enum):
    MID_PRICE = "midprice"
    MID_PRICE_FUTURE = "m+t"
    MID_PRICE_PAST = "m-t"

    PREDICTION = "y"
    PERCENTAGE_CHANGE = "ratio_y"

    L1_BUY_PRICE = 'pbuy1'
    L1_SELL_PRICE = 'psell1'


def add_lob_labels(data, window_size_forward, window_size_backward, label_threshold, sigma_fraction):
    """ Labels the data in [0, 1, 2], labels 0 (down), 1 (stable), 2 (down). """
    data = add_midprices_columns(data, window_size_forward, window_size_backward)

    # we remove the first and the last rolling_tu because the rolling mean has holes
    data = data[(window_size_backward - 1):-(window_size_forward - 1)]

    data[DataCols.PERCENTAGE_CHANGE.value] = get_percentage_change(data, DataCols.MID_PRICE_PAST.value, DataCols.MID_PRICE_FUTURE.value)
    ratio_mu, ratio_si = data[DataCols.PERCENTAGE_CHANGE.value].mean(), data[DataCols.PERCENTAGE_CHANGE.value].std()

    # pd.DataFrame(data[DataCols.PERCENTAGE_CHANGE.value]).hist(bins=100)
    # plt.show()

    label_threshold = (ratio_mu + ratio_si * sigma_fraction) if sigma_fraction is not None else label_threshold

    # labels 0 (down), 1 (stable), 2 (down)
    data[DataCols.PREDICTION.value] = np.ones(data.shape[0])
    data[DataCols.PREDICTION.value] = np.where(data[DataCols.PERCENTAGE_CHANGE.value] > label_threshold, 2, data[DataCols.PREDICTION.value])
    data[DataCols.PREDICTION.value] = np.where(data[DataCols.PERCENTAGE_CHANGE.value] < -label_threshold, 0, data[DataCols.PREDICTION.value])
    return data, label_threshold


def add_midprices_columns(data, window_size_forward, window_size_backward):
    data[DataCols.MID_PRICE.value] = get_mid_price(data)
    data[DataCols.MID_PRICE_FUTURE.value] = data[DataCols.MID_PRICE.value].rolling(window_size_forward).mean().shift(- window_size_forward + 1)
    data[DataCols.MID_PRICE_PAST.value] = data[DataCols.MID_PRICE.value].rolling(window_size_backward).mean()
    return data


def plot_dataframe_stats(data, label_threshold):
    """ Plots the predictions Y and histogram. Plots mid-price and shifted averages. """
    data = data.reset_index()

    ax = data[[DataCols.MID_PRICE.value,
               DataCols.MID_PRICE_FUTURE.value,
               DataCols.MID_PRICE_PAST.value]].plot()

    yticks, _ = plt.yticks()

    ax2 = ax.twinx()
    ax2 = data[[DataCols.PERCENTAGE_CHANGE.value]].plot(ax=ax)
    ax2.set_ylim([-1, max(yticks)])
    ax2.axhline(y= label_threshold, color='r', linestyle=':')
    ax2.axhline(y=-label_threshold, color='r', linestyle=':')
    data[[DataCols.PREDICTION.value]].plot()
    data[[DataCols.PREDICTION.value]].hist()
    plt.show()


def get_mid_price(data):
    return (data[DataCols.L1_BUY_PRICE.value] + data[DataCols.L1_SELL_PRICE.value]) / 2


def get_percentage_change(data, past, future):
    return (data[future] - data[past]) / data[past]


def get_volume_column_name(columns):
    return [f for f in columns if "vbuy" in f or "vsell" in f]


def get_price_column_name(columns):
    return [f for f in columns if "pbuy" in f or "psell" in f]


def get_sell_column_name(columns):
    return [f for f in columns if "vsell" in f or "psell" in f]


def get_buy_column_name(columns):
    return [f for f in columns if "pbuy" in f or "vbuy" in f]
