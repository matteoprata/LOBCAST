
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
    """ DONE: remember to use the mean/std of the training set, to z-normalize the test set. """

    col_choice = {"volumes": get_volume_column_name(data.columns),
                  "prices":  get_price_column_name(data.columns)}

    print("Normalization... (using means", normalization_mean, "and stds", normalization_std, ")")

    means_dict, stds_dict = dict(), dict()
    for col_name in col_choice:
        cols = col_choice[col_name]

        if normalization_mean is None and normalization_std is None:
            means_dict[col_name] = data.loc[:, cols].stack().mean()
            stds_dict[col_name] = data.loc[:, cols].stack().std()

        elif normalization_mean is not None and normalization_std is not None:
            means_dict[col_name] = normalization_mean[col_name]
            stds_dict[col_name] = normalization_std[col_name]

        data.loc[:, cols] = (data.loc[:, cols] - means_dict[col_name]) / stds_dict[col_name]
        data.loc[:, cols] = data.loc[:, cols]

        # TODO: volumes and prices can be negative, add min value
        # + abs(data.loc[:, cols].stack().min())  # scale positive
        # data.loc[:, cols].stack().plot.hist(bins=200, alpha=0.5, title=col_name)
        # plt.show()

    data = data.fillna(method="bfill")
    data = data.fillna(method="ffill")
    return data, means_dict, stds_dict


class DataCols(Enum):
    MID_PRICE = "midprice"
    MID_PRICE_FUTURE = "m+t"
    MID_PRICE_PAST = "m-t"

    PREDICTION = "y"
    PERCENTAGE_CHANGE = "ratio_y"

    L1_BUY_PRICE = 'pbuy1'
    L1_SELL_PRICE = 'psell1'


def add_lob_labels(data, window_size_forward, window_size_backward, label_threshold_pos, label_threshold_neg, sigma_fraction):
    """ Labels the data in [0, 1, 2], labels 0 (down), 1 (stable), 2 (down). """
    data = add_midprices_columns(data, window_size_forward, window_size_backward)  # MID_PRICE, MID_PRICE_FUTURE, MID_PRICE_PAST

    # we remove the first and the last rolling_tu because the rolling mean has holes
    data = data[(window_size_backward - 1):-(window_size_forward - 1)]  # ok checked 6-03-23

    data[DataCols.PERCENTAGE_CHANGE.value] = get_percentage_change(data, DataCols.MID_PRICE_PAST.value, DataCols.MID_PRICE_FUTURE.value)
    ratio_mu, ratio_si = data[DataCols.PERCENTAGE_CHANGE.value].mean(), data[DataCols.PERCENTAGE_CHANGE.value].std()

    # pd.DataFrame(data[DataCols.PERCENTAGE_CHANGE.value]).hist(bins=100)
    # plt.show()

    label_threshold_pos = (ratio_mu + ratio_si * sigma_fraction) if sigma_fraction is not None else label_threshold_pos
    label_threshold_neg = (ratio_mu - ratio_si * sigma_fraction) if sigma_fraction is not None else label_threshold_neg

    # labels 0 (down), 1 (stable), 2 (up)
    data[DataCols.PREDICTION.value] = np.ones(data.shape[0])
    data[DataCols.PREDICTION.value] = np.where(data[DataCols.PERCENTAGE_CHANGE.value] > label_threshold_pos, 2, data[DataCols.PREDICTION.value])
    data[DataCols.PREDICTION.value] = np.where(data[DataCols.PERCENTAGE_CHANGE.value] < label_threshold_neg, 0, data[DataCols.PREDICTION.value])

    return data, label_threshold_pos, label_threshold_neg


def add_lob_labels_march_2023(data, window_size_forward, window_size_backward, alpha):
    """ Labels the data in [0, 1, 2], labels 0 (down), 1 (stable), 2 (down). """
    data = add_midprices_columns(data, window_size_forward, window_size_backward)  # MID_PRICE, MID_PRICE_FUTURE, MID_PRICE_PAST

    # we remove the first and the last rolling_tu because the rolling mean has holes
    data = data[(window_size_backward - 1):-window_size_forward]

    data[DataCols.PERCENTAGE_CHANGE.value] = get_percentage_change(data, DataCols.MID_PRICE_PAST.value, DataCols.MID_PRICE_FUTURE.value)

    label_threshold_pos = alpha
    label_threshold_neg = - alpha

    # labels 0 (down), 1 (stable), 2 (up)
    data[DataCols.PREDICTION.value] = np.ones(data.shape[0])
    data[DataCols.PREDICTION.value] = np.where(data[DataCols.PERCENTAGE_CHANGE.value] > label_threshold_pos, 2, data[DataCols.PREDICTION.value])
    data[DataCols.PREDICTION.value] = np.where(data[DataCols.PERCENTAGE_CHANGE.value] < label_threshold_neg, 0, data[DataCols.PREDICTION.value])

    return data


def add_midprices_columns(data, window_size_forward, window_size_backward):
    data[DataCols.MID_PRICE.value] = get_mid_price(data)
    data[DataCols.MID_PRICE_FUTURE.value] = data[DataCols.MID_PRICE.value].rolling(window_size_forward).mean().shift(-window_size_forward)
    data[DataCols.MID_PRICE_PAST.value]   = data[DataCols.MID_PRICE.value].rolling(window_size_backward).mean()

    return data


def plot_dataframe_stats(data, label_threshold_pos, label_threshold_neg, dataset_type):
    """ Plots the predictions Y and histogram. Plots mid-price and shifted averages. """
    data = data.reset_index()

    ax = data[[DataCols.MID_PRICE.value,
               DataCols.MID_PRICE_FUTURE.value,
               DataCols.MID_PRICE_PAST.value]].plot()

    yticks, _ = plt.yticks()

    ax2 = ax.twinx()
    ax2 = data[[DataCols.PERCENTAGE_CHANGE.value]].plot(ax=ax)
    ax2.set_ylim([-1, max(yticks)])
    ax2.axhline(y=label_threshold_pos, color='red', linestyle=':')
    ax2.axhline(y=label_threshold_neg, color='blue', linestyle=':')
    #data[[DataCols.PREDICTION.value]].plot()
    data[[DataCols.PREDICTION.value]].hist()
    plt.title(dataset_type)
    # plt.show()


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
