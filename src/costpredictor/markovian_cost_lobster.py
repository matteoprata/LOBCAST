import pandas as pd
import numpy as np
from tqdm import tqdm
from src import config
import os

from matplotlib import pyplot as plt
import src.utils.lob_util as lbu

pd.set_option('display.float_format', lambda x: '%.4f' % x)
#pd.set_option("display.max_rows", None, "display.max_columns", None)


# TODO: lambdone calculation rate/unit.
# TODO: come passare da 4x4 matrix a 10x10 matrix that we need. (mattew)
# TODO: plot the mean/std at varying stage of the day for each level
# TODO: discretizzare in 4 quartili l'outstading volume e std; e poi per vedere in ogni secondo
#  quanti ordini per ogni elem in (4x4) vegono eseguiti --> rate


def probability_exec(rates : np.ndarray, delta : int = 1.02):
    """ it takes in input the nd array with rates and compute the cumulative distr funtction probability

        delta is the expected life time for an order inside the orderbook.
        We would like to know the expected probability of its execution
    """
    f = lambda x : 1 - np.e**(-(delta*x))
    prob_values = f(rates)
    return prob_values


def compute_probability_rates(df : pd.DataFrame, debug : bool = False) -> np.ndarray:
    """
        We consider unit of time 1 second
        Compute the quantiles and for each combination the rate of execution for orders in that unit of time
    """
    # compute quantiles
    volatilies_quantiles = list(compute_quartiles(df, "volatility"))
    volume_quantiles = list(compute_quartiles(df, "outstanding_volume"))

    # add infinite to the quantiles to speed up and simplify computation
    volatilies_quantiles += [np.inf]
    volume_quantiles += [np.inf]

    if debug:
        print(volatilies_quantiles)
        print(volume_quantiles)

    # make sure we have correct time here
    df["time_to_fill"] = pd.to_timedelta(df["time_to_fill"])

    # volatility on columns
    # [[vol <= q1, q1 < vol <= q2, ... , q3 < vol] (0 <= volatiliy <= q1)
    # [vol <= q1, q1 < vol <= q2, ..., q3 < vol] (q1 < volatiliy <= q2)
    # [vol <= q1, q1 < vol <= q2, ..., q3 < vol] (q2 < volatiliy <= q3)
    # [vol <= q1, q1 < vol <= q2, ..., q3 < vol] (q3 < volatiliy)
    # ]
    rates_array = np.zeros((4, 4))
    for i in range(len(volatilies_quantiles)):
        for j in range(len(volume_quantiles)):
            # compute rate execution
            # e.g., for i == 0 we have volatility_value < 0.25 quantile
            # e.g., for i == 1 we have volatility_value < 0.5 quantile
            # ...
            # e.g., for i == 3 we have volatility_value < np.inf
            cond_volatility = df["volatility"] <= volatilies_quantiles[i]
            cond_volume = df["outstanding_volume"] <= volume_quantiles[j]

            if debug:
                print("volatile <=", volatilies_quantiles[i], " - LEN:", len(df[cond_volatility]))
                print("volume <=", volume_quantiles[j], " - LEN:", len(df[cond_volume]))

            # add lower bound
            if i > 0:
                # e.g., for i == 1 we have volatility_value > 0.25 quantile
                cond_volatility = cond_volatility & (df["volatility"] > volatilies_quantiles[i - 1])
                if debug:
                    print(volatilies_quantiles[i - 1], " < volatile <=", volatilies_quantiles[i], " - LEN:",
                          len(df[cond_volatility]))
            if j > 0:
                cond_volume = cond_volume & (df["outstanding_volume"] > volume_quantiles[j - 1])
                if debug:
                    print(volume_quantiles[j - 1], "< volume <=", volume_quantiles[j], " - LEN:", len(df[cond_volume]))

            if debug:
                print("all-cond", len(df[cond_volume & cond_volatility]))
                print("------")

            sub_df = df[cond_volatility & cond_volume]
            mean_time_to_fill = sub_df["time_to_fill"].mean().seconds
            rates_array[i, j] = 1 / mean_time_to_fill
            if debug:
                print("volatility - volume ", volatilies_quantiles[i], volume_quantiles[j], " pre-prob:", rates_array[i, j])

    return rates_array, volatilies_quantiles[:-1], volume_quantiles[:-1]


def compute_quartiles(df: pd.DataFrame, column : str):
    """
        The method compute the 4 quartiles for the values in the given column.

        df : the dataframe with the orders executions, their time_to_fill, outstanding_volume, ...
        column : the column of which we want to compute the 4 quartiles
    """
    q1 = df[column].quantile(q=0.25, interpolation='linear')
    q2 = df[column].quantile(q=0.5, interpolation='linear')
    q3 = df[column].quantile(q=0.75, interpolation='linear')

    return q1, q2, q3


def add_outstanding_volume(df : pd.DataFrame, perc_volumes : bool = False) -> pd.DataFrame:
    """ This method returns a new dataframe with the outstanding volume in each level,
        instead of the simple level volume

        perc_volumes : work on perc volumes instead of integer volumes
    """
    df = df.copy()

    # first, we change the volumes on the df, using a cumulative version to achieve the outstanding volumes
    max_lev_column_sell, sell_v_columns = "vsell10", [f for f in df.columns if "vsell" in f]
    max_lev_column_buy, buy_v_columns = "vbuy10", [f for f in df.columns if "vbuy" in f]

    df[sell_v_columns] = df[sell_v_columns].cumsum(axis=1)
    df[buy_v_columns] = df[buy_v_columns].cumsum(axis=1)

    # TODO: we do not use percentange of volumes
    # then we compute the percentage of volume in each lev, respect the overall volume in the 10 levels.
    if perc_volumes:
        df[sell_v_columns] = df[sell_v_columns].div(df[max_lev_column_sell], axis=0)
        df[buy_v_columns] = df[buy_v_columns].div(df[max_lev_column_buy], axis=0)

    return df


def add_volatility(df : pd.DataFrame, volatility_window : str = "1h", min_granularity : str = "s") -> pd.DataFrame:
    """ Add the volatility inside the orderbook df, using the mid-price

        min_granularity = the granularity of the minimum step inside the orderbook
        volatility_window: the window to use in computing the volatility

        return None
    """
    # resample at a correct granularity
    new_df = df[["mid_price"]]
    new_df = new_df.resample("s").last()  # close value

    # compute volatility
    new_df["volatility"] = new_df.rolling(volatility_window).std()
    new_df = new_df.reset_index().rename(columns={"date" : "new_date"})
    new_df = new_df.fillna(method="bfill")
    new_df.drop(columns=["mid_price"], inplace=True)

    # reset index and put again the data inside df
    df.reset_index(inplace=True)
    df["new_date"] = df["date"].dt.floor("S")

    df = df.merge(new_df, on="new_date")
    df.drop(columns=["new_date"], inplace=True)
    df.set_index("date", inplace=True)
    return df


def compute_cost_matrix(data_path : str, debug : bool = False, save_intermediate_file : bool = True,
                        volatility_window : str = "1h", mid_price_approach : bool = True):
    """
        data_path : path to the lobster data to load
        debug: whether print/plot debug info
        save_intermediate_file : whether save or not intermediate compute files
        volatility_window : the window to use in order to get the volatily of the stock for a given order
        mid_price_approach : when enabled, the mid-price approach computes the execution of an order according the current value of the mid price.
    """
    # read files
    df = lbu.from_folder_to_unique_df(data_path, plot=False, level=10, granularity=None, add_messages=True)
    # try to infer time to till for each new submitted order in the buy or sell side
    side_to_analyse = 1  # buy orders, this means we skip sell orders from the analysis

    # this would be the final results that we are going to use to compute the probabilities and time to fill
    # inside the GAN
    time_to_fill_df = {"order_id" : [], "execution_time" : [], "submission_time" : [],
                       "level" : [], "outstanding_volume" : [], "time_to_fill" : [],
                       "side" : [], "price" : [], "volatility" : []}

    df = add_outstanding_volume(df)

    # add mid-price for matches
    df["mid_price"] = (df["psell1"] + df["pbuy1"]) / 2

    # Add volatility feature to the dataframe
    df = add_volatility(df, volatility_window=volatility_window)

    if debug:
        df = df["mid_price"]
        df = df.resample("1s").last()
        df = df.fillna(method="ffill")
        print(df)
        df.plot()
        plt.show()

    # Find matches
    # for each submitted order we keep a dictionary with its time, outstanding volume, and price.
    # after each iteration, when we found an execution to this price or lower (higher) for a buy (sell)
    # we do a match and save the times.
    # TODO: we are not interested in market orders right? but maybe write something on why on the tex

    orders_to_match = []
    for i, r in tqdm(df.iterrows(), total=df.shape[0]):

        # new submission here
        if r["event_type"] == config.OrderEvent.SUBMISSION.value:
            if r["direction"] != side_to_analyse:
                continue  # we are not interested in the other side here

            # compute level of the order
            side_key = "buy" if side_to_analyse == 1 else "sell"
            for l in range(1, 11):
                # buy case (side == 1) if the submission price > level_price I should be in the previous level.
                # higher first in this side
                if r["p" + side_key + str(l)] < r["price"] and side_to_analyse == 1:
                    break
                elif r["p" + side_key + str(l)] > r["price"] and side_to_analyse == -1:
                    break

                # TODO: change in a more elegant from.
                #  the point is that: if we don't break at the last level, 10,
                # this means that we are exactly at the last level and level_of_order = 10, not 9
                # this is a way to handle this in bad
                if l == 10:
                    l = 11

            level_of_order = l - 1  # because we stop exactly at the next best level
            start_time = i
            if level_of_order == 0:  # immediate execution
                time_to_fill_df["order_id"] += [r["order_id"]]
                time_to_fill_df["execution_time"] += [i]
                time_to_fill_df["submission_time"] += [i]
                time_to_fill_df["level"] += [0]
                time_to_fill_df["outstanding_volume"] += [0]
                time_to_fill_df["time_to_fill"] += [0]
                time_to_fill_df["side"] += [side_to_analyse]
                time_to_fill_df["price"] += [r["price"]]
                time_to_fill_df["volatility"] += [r["volatility"]]
            else:
                outstanding_volume = r["v" + side_key + str(level_of_order)] - r["size"]
                assert outstanding_volume >= 0
                orders_to_match.append({"submission_time" : i, "outstanding_volume" : outstanding_volume, "level" : level_of_order,
                                        "order_id" : r["order_id"], "price" : r["price"]})

        if mid_price_approach:
            new_orders_to_match = []
            for otm in orders_to_match:
                # FIRST CLAUSE:
                # someone has sold its stock, and this match a buy order!
                # if they sold at a lower price than my buy order, I should buy
                # SECOND CLAUSE:
                # on the other hand, if there is a mathc in the sell side,
                # if the execeute price is greater than my price i should execute
                if ((r["mid_price"] <= otm["price"] and side_to_analyse == 1) or
                        (r["mid_price"] >= otm["price"] and side_to_analyse == -1)):
                    time_to_fill_df["order_id"] += [otm["order_id"]]
                    time_to_fill_df["execution_time"] += [i]
                    time_to_fill_df["submission_time"] += [otm["submission_time"]]
                    time_to_fill_df["level"] += [otm["level"]]
                    time_to_fill_df["outstanding_volume"] += [otm["outstanding_volume"]]
                    time_to_fill_df["time_to_fill"] += [np.nan]
                    time_to_fill_df["side"] += [side_to_analyse]
                    time_to_fill_df["price"] += [otm["price"]]
                    time_to_fill_df["volatility"] += [r["volatility"]]
                else:
                    new_orders_to_match.append(otm)

            orders_to_match = new_orders_to_match
        else:
            # find matches and filling times
            if r["event_type"] in [config.OrderEvent.EXECUTION.value, config.OrderEvent.HIDDEN_EXECUTION.value]:
                executed_volume = r["size"]
                executed_price = r["price"]
                executed_direction = r["direction"]
                new_orders_to_match = []
                if side_to_analyse == executed_direction:
                    continue  # we are not interested in executions that affect the order side

                for otm in orders_to_match:
                    # we already matched all we can, no further orders can be matched
                    if executed_volume <= 0:
                        new_orders_to_match.append(otm)
                        continue

                    # FIRST CLAUSE:
                    # someone has sold its stock, and this match a buy order!
                    # if they sold at a lower price than my buy order, I should buy
                    # SECOND CLAUSE:
                    # on the other hand, if there is a mathc in the sell side,
                    # if the execeute price is greater than my price i should execute
                    if ((executed_price <= otm["price"] and side_to_analyse == 1) or
                            (executed_price >= otm["price"] and side_to_analyse == -1)):
                        executed_volume -= otm["outstanding_volume"]
                        time_to_fill_df["order_id"] += [otm["order_id"]]
                        time_to_fill_df["execution_time"] += [i]
                        time_to_fill_df["submission_time"] += [otm["submission_time"]]
                        time_to_fill_df["level"] += [otm["level"]]
                        time_to_fill_df["outstanding_volume"] += [otm["outstanding_volume"]]
                        time_to_fill_df["time_to_fill"] += [np.nan]
                        time_to_fill_df["side"] += [side_to_analyse]
                        time_to_fill_df["price"] += [otm["price"]]
                        time_to_fill_df["volatility"] += [r["volatility"]]
                    else:
                        new_orders_to_match.append(otm)

                orders_to_match = new_orders_to_match

    final_df = pd.DataFrame(time_to_fill_df)
    final_df["time_to_fill"] = final_df["execution_time"] - final_df["submission_time"]
    final_df["level"].unique()

    if save_intermediate_file:
        final_df.to_csv("fill_times.csv", index=False)

    # final plot to make sure of stuff
    if debug:
        df = df["mid_price"]
        df = df.resample("1s").last()
        df = df.fillna(method="ffill")
        print(df)
        df.plot()
        plt.show()

    return df

def compute_npy_cost_matrices(input_dir):
    stock_dir = input_dir.replace("/", "_")

    out_dir = "data/thresholds/" +stock_dir + "/"
    os.makedirs(out_dir, exist_ok=True)

    df = compute_cost_matrix(input_dir, save_intermediate_file=False, debug=False)
    # reload it from a previous version
    #df = pd.read_csv("fill_times.csv")
    # compute probabilities
    prob_rates, volatilies_quantiles, volume_quantiles = compute_probability_rates(df)
    prob_exec = probability_exec(prob_rates)

    filename = out_dir + "prob_exec.npy"
    with open(filename, 'wb') as f:
        np.save(f, prob_exec)

    filename = out_dir + "volatilies_quantiles.npy"
    with open(filename, 'wb') as f:
        np.save(f, volatilies_quantiles)

    filename = out_dir + "volume_quantiles.npy"
    with open(filename, 'wb') as f:
        np.save(f, volume_quantiles)

if __name__ == "__main__":
    # compute the df with order executions
    # df = compute_cost_matrix("data/AMZN_Sample")
    # reload it from a previous version

    df = pd.read_csv("fill_times.csv")
    # compute probabilities
    prob_rates, volatilies_quantiles, volume_quantiles = compute_probability_rates(df)
    prob_exec = probability_exec(prob_rates)

    filename = "prob_exec.npy"
    with open(filename, 'wb') as f:
        np.save(f, prob_exec)

    filename = "volatilies_quantiles.npy"
    with open(filename, 'wb') as f:
        np.save(f, volatilies_quantiles)

    filename = "volume_quantiles.npy"
    with open(filename, 'wb') as f:
        np.save(f, volume_quantiles)

    with open(filename, 'rb') as f:
        prob_exec = np.load(f)

    print(prob_exec)
