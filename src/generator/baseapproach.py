# ------------------- BASE APPROACH TO ADVERSARIAL ATTACKS (UN-Targeted) -------------------
import numpy as np
import pandas as pd

from enum import Enum
from matplotlib import pyplot as plt
from matplotlib import colors
from datetime import datetime
from plotly import graph_objs as go


# ---- Pollution system ----#
class PollutionType(Enum):
    SELL_ONLY = ("sell", 1)
    BUY_ONLY = ("buy", 1)
    BOTH_SELL_BUY = ("mix", 2)


# basic paramenters to pollution
ORDER_EVERY_TU = 15 # parametro da variare
ORDERS_DURATION_TU = 15
N_ORDERS_TU = 5 # parametro da variare
MAX_LEVEL_POLLUTION = 10
POLLUTION_TYPE = PollutionType.BOTH_SELL_BUY 
BASE_PERCENTAGE = 0.1
PERCENTAGES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5] # desidered


def pollute_training_df(df: pd.DataFrame, order_every_tu: int, orders_duration_tu: int, n_orders_tu: int, pollution_type: PollutionType, n_levels: int, default_lvl_placement: list=None, seed: int=0):
    
    assert(n_orders_tu <= n_levels-2)
    assert(default_lvl_placement is None or len(default_lvl_placement) == n_orders_tu)
    np.random.seed(seed)

    # if mix do it both for sell and buy
    if pollution_type == PollutionType.BOTH_SELL_BUY:
        for pt in [PollutionType.SELL_ONLY, PollutionType.BUY_ONLY]:
            df = __pollution(df, order_every_tu, orders_duration_tu, n_orders_tu, pt, n_levels, default_lvl_placement)
    else:
        df = __pollution(df, order_every_tu, orders_duration_tu, n_orders_tu, pollution_type, n_levels, default_lvl_placement)
    return df


def __pollution(df: pd.DataFrame, order_every_tu: int, orders_duration_tu: int, n_orders_tu: int, pollution_type: PollutionType, n_levels: int, default_lvl_placement: list=None):
    lvl_placement = sorted(default_lvl_placement) if default_lvl_placement is not None else None

    # columns interested for the shifts and insertions, e.g. psell, vsell if pollution_type is sell
    cols_type = []
    for i in range(1, n_levels + 1):
        cols_type += ["p{}{}".format(pollution_type.value[0], i), "v{}{}".format(pollution_type.value[0], i)]

    df_type = df.loc[:, cols_type]  # isolate the columns of interest

    # get minimum volume to substitute 
    vcols = [v for v in df.columns if "v" in v]
    min_volume = min(df.loc[:, vcols].min().min(), 1)  # to handle lobster data

    cons_block = 0
    for i in range(df_type.shape[0]):                     # for every row
        if i % order_every_tu == 0 or cons_block > 0:     # do the shift and adjust the prices for a block of orders_duration_tu seconds
            cons_block += 1

            # list containing the indices of the insertion in N.B. [1,n_levels-1], e.g. [1,5,6] makes insetions between levels 0-1, 4-5, 5-6
            # they are random for every block, or preassigned from the input with default_lvl_placement

            is_compute_placement = i % order_every_tu == 0 and default_lvl_placement is None
            lvl_placement = sorted(np.random.randint(low=1, high=n_levels - 1, size=n_orders_tu)) if is_compute_placement else lvl_placement

            for lvl_pl in lvl_placement:
                prev_val = df_type.iloc[i, 2 * lvl_pl - 2]                                                                # value to increment/decrement (depending if sell or buy)
                # NOTE: lobster data has only integer, so basically if you have L1 32828 and L2 32827 --> 32827.5 
                # TODO: add a control, if we can really add a new level, or the two levels are too tight
                price_delta = (df_type.iloc[i, 2 * lvl_pl] - df_type.iloc[i, 2 * lvl_pl - 2]) / 2                         # quantity for the increment
                
                df_type.iloc[i:i + 1, 2 * lvl_pl:] = df_type.iloc[i:i + 1, 2 * lvl_pl:].shift(periods=2, axis="columns")  # do shift
                df_type.iloc[i, 2 * lvl_pl: 2 * lvl_pl + 2] = [prev_val + price_delta, min_volume]                                 # do fill

            if cons_block == orders_duration_tu:  # block of fake orders is done
                cons_block = 0

    df.loc[:, cols_type] = df_type.loc[:, cols_type]
    return df


# ---- perturbation percentage ---- #

def calculate_percentage(df_len: int, order_every_tu: int, orders_duration_tu: int, n_orders_tu: int, pollution_type: int, n_levels: int):
    return (df_len / order_every_tu * orders_duration_tu * n_orders_tu * pollution_type) / (df_len * n_levels * pollution_type)

def calculate_percentage_simplified(order_every_tu: int, orders_duration_tu: int, n_orders_tu: int, n_levels: int):
    return orders_duration_tu * n_orders_tu / (order_every_tu * n_levels)

def calculate_perturbation_frequence(percentage: int, orders_duration_tu: int, n_orders_tu: int, n_levels: int):
    return orders_duration_tu * n_orders_tu / (percentage * n_levels)

def calculate_perturbation_levels(percentage: int, orders_duration_tu: int, order_every_tu: int, n_levels: int):
    return  (percentage * n_levels *  order_every_tu) / orders_duration_tu

def manual_pollution(lob_df : pd.DataFrame, seed : int = 1, order_every_tu=None, 
                percentage_base_of_pollution=BASE_PERCENTAGE,
                orders_duration_tu=ORDERS_DURATION_TU, n_orders_tu=N_ORDERS_TU, 
                pollution_type=POLLUTION_TYPE, max_levels=MAX_LEVEL_POLLUTION):
    """ Apply a manual pollution to the input df 

    Args:

        lob_df (pd.DataFrame): the dataframe with 40+1 columns and n rows
        seed (int, optional): Control the random pollution . Defaults to 1.
        order_every_tu ([type], optional): The frequency (slices) before putting new pollution. Defaults to f.
        percentage_base_of_pollution (float): The percentage of orderbook to polluted
        orders_duration_tu ([type], optional): The time a pollution will stay in the orderbook, before beeing removed. Defaults to Dataset.ORDERS_DURATION_TU.
        n_orders_tu ([type], optional): The number of polluted orders to place. Defaults to Dataset.N_ORDERS_TU.
        pollution_type ([type], optional): which side of the orderbook to pollude. Defaults to Dataset.POLLUTION_TYPE.
        max_levels ([type], optional): The maximum number of levels to pollude. Defaults to MAX_LEVEL_POLLUTION.

    Returns:
        out_df: the polluded pd.DataFrame 
    """
    lob_df = lob_df.copy()
    
    if order_every_tu is None:
        order_every_tu = calculate_perturbation_frequence(percentage_base_of_pollution, orders_duration_tu, n_orders_tu, max_levels)
    
    return pollute_training_df(df=lob_df, order_every_tu=order_every_tu, orders_duration_tu=orders_duration_tu, 
                n_orders_tu=n_orders_tu, pollution_type=pollution_type, n_levels=max_levels, seed=seed) 


def multiple_increasing_pollution_order_every_tu(lob_df : pd.DataFrame, percentages_of_pollution=PERCENTAGES, seed=1,
                orders_duration_tu=ORDERS_DURATION_TU, n_orders_tu=N_ORDERS_TU, order_every_tu=ORDER_EVERY_TU,
                pollution_type=POLLUTION_TYPE, max_levels=MAX_LEVEL_POLLUTION):
    """ The method is used to produce several instances of dataset with different pollutions

    Args:
        lob_df (pd.DataFrame): the dataframe with 40+1 columns and n rows
        order_every_tu ([type], optional): The frequency (slices) before putting new pollution. Defaults to f.
        percentages_of_pollution ([type], optional): The percentage of pollution to have. Defaults to Dataset.PERCENTAGES.
        seed (int, optional): Control the random pollution . Defaults to 1.
        orders_duration_tu ([type], optional): The time a pollution will stay in the orderbook, before beeing removed. Defaults to Dataset.ORDERS_DURATION_TU.
        n_orders_tu ([type], optional): The number of polluted orders to place. Defaults to Dataset.N_ORDERS_TU.
        pollution_type ([type], optional): which side of the orderbook to pollude. Defaults to Dataset.POLLUTION_TYPE.
        max_levels ([type], optional): The maximum number of levels to pollude. Defaults to MAX_LEVEL_POLLUTION.


        return a list of [(percentage_of_pollution, order_every_tu, polluted_df)]
    """
    # at different frequencies
    df_at_dif_freq = {}
    for p_o_p in percentages_of_pollution:
        order_every_tu = calculate_perturbation_frequence(p_o_p, orders_duration_tu, n_orders_tu, max_levels)
        yield (p_o_p, order_every_tu, manual_pollution(lob_df, percentage_base_of_pollution=p_o_p, orders_duration_tu=orders_duration_tu,
                                                            n_orders_tu=n_orders_tu, pollution_type=pollution_type, max_levels=max_levels,
                                                            order_every_tu=None))
        


def multiple_increasing_pollution_n_orders_tu(lob_df : pd.DataFrame, percentages_of_pollution=PERCENTAGES, seed=1,
                orders_duration_tu=ORDERS_DURATION_TU, n_orders_tu=N_ORDERS_TU, order_every_tu=ORDER_EVERY_TU,
                pollution_type=POLLUTION_TYPE, max_levels=MAX_LEVEL_POLLUTION):
    """ The method is used to produce several instances of dataset with different pollutions based on percentage of pollution

        Upon the percentage of pollution we change the n_orders_tu.

    Args:
        lob_df (pd.DataFrame): the dataframe with 40+1 columns and n rows
        order_every_tu ([type], optional): The frequency (slices) before putting new pollution. Defaults to f.
        percentages_of_pollution ([type], optional): The percentage of pollution to have. Defaults to Dataset.PERCENTAGES.
        seed (int, optional): Control the random pollution . Defaults to 1.
        orders_duration_tu ([type], optional): The time a pollution will stay in the orderbook, before beeing removed. Defaults to Dataset.ORDERS_DURATION_TU.
        n_orders_tu ([type], optional): The number of polluted orders to place. Defaults to Dataset.N_ORDERS_TU.
        pollution_type ([type], optional): which side of the orderbook to pollude. Defaults to Dataset.POLLUTION_TYPE.
        max_levels ([type], optional): The maximum number of levels to pollude. Defaults to MAX_LEVEL_POLLUTION.

        return a list of [(percentage_of_pollution, n_orders_tu, polluted_df)]
    """
    # at different number of orders 
    for p_o_p in percentages_of_pollution:
        notu = calculate_perturbation_levels(p_o_p, orders_duration_tu, order_every_tu, max_levels)
        notu = int(notu)
        yield (p_o_p, notu, self.manual_pollution(lob_df, orders_duration_tu=orders_duration_tu, n_orders_tu=notu, 
                                                            pollution_type=pollution_type, max_levels=max_levels,
                                                            order_every_tu=order_every_tu))
    


