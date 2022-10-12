import pandas as pd
import numpy as np
import os

from src.costpredictor import markovian_cost_lobster as mcl
from src.utils.lobdataset import DEEPDataset

pd.set_option('display.float_format', lambda x: '%.4f' % x)

# here for dataset FI we compute the volatility and outstading volumes to then
# compute the 4x4 (volatility, volumes) execution probabilities

def compute_fi_matrices(df : pd.DataFrame, is_buy_side=1, volatility_window : int = 3600, data_dir="",
                        debug = True) -> None:
    """
    The method should be called at the beginning of the GAN execution,
        to compute the probability to execute a new order with a given outstading volume and volatility

    :param df: the input dataset with prices and volume levels
    :param is_buy_side: the side to analyse, 1 for True
    :return:
    """
    side_key = "buy" if is_buy_side else "sell"

    # compute outstanding volumes
    df = mcl.add_outstanding_volume(df)

    # add mid-price for matches
    df["mid_price"] = (df["psell1"] + df["pbuy1"]) / 2

    # Add volatility feature to the dataframe
    df["volatility"] = df["mid_price"].rolling(volatility_window).std()

    # compute quantiles
    volatilies_quantiles = list(mcl.compute_quartiles(df, "volatility"))
    # stack outstanding volumes
    vol_columns = [x for x in df.columns if "v" + side_key in x]
    df_volumes = pd.concat([df[[col]].rename(columns={col : "outstanding_volume"}) for col in vol_columns], axis=0)
    volume_quantiles = list(mcl.compute_quartiles(df_volumes, "outstanding_volume"))

    # save quantiles
    filename = data_dir + "volatilies_quantiles.npy"
    os.makedirs(data_dir, exist_ok=True)
    with open(filename, 'wb') as f:
        np.save(f, volatilies_quantiles)

    filename = data_dir + "volume_quantiles.npy"
    with open(filename, 'wb') as f:
        np.save(f, volume_quantiles)

    # add infinite to the quantiles to speed up and simplify computation
    volatilies_quantiles += [np.inf]
    volume_quantiles += [np.inf]

    # compute exec-probabilities
    # shifted mid-price to compute execution and not-execution events, in the following second.
    # we remember that FI is 1-sec granularity, and we are interested in computing the probabilty
    # that an order places on a given (volume, price) volatiliyy is execute within the next second
    # so we can just count the number of executions in the following row.
    df["exec_midprice"] = df["mid_price"].shift(-1)

    colums_to_anlyse = [x for x in df.columns if "p" + side_key in x]
    for col in colums_to_anlyse:
        if is_buy_side:
            df[col] = (df["exec_midprice"] <= df[col])*1
        else:
            df[col] = (df["exec_midprice"] >= df[col])*1

    prob_array = np.zeros((4, 4))
    for i in range(len(volatilies_quantiles)):
        for j in range(len(volume_quantiles)):
            # compute rate execution

            cond_volatility = df["volatility"] <= volatilies_quantiles[i]
            # add lower bound
            if i > 0:
                # e.g., for i == 1 we have volatility_value > 0.25 quantile
                cond_volatility = cond_volatility & (df["volatility"] > volatilies_quantiles[i - 1])
                if debug:
                    print(volatilies_quantiles[i - 1], " < volatile <=", volatilies_quantiles[i], " - LEN:",
                          len(df[cond_volatility]))

            df_volatiliy = df.loc[cond_volatility]

            count_events = 0
            favorable_events = 0
            for l in range(1, 11):
                column = "v" + side_key + str(l)
                match_df = df_volatiliy[df_volatiliy[column] <= volume_quantiles[j]]
                if j > 0:
                    match_df = match_df[match_df > volume_quantiles[j - 1]]

                if len(match_df) > 0:
                    count_events += len(match_df)
                    favorable_events += match_df["p" + side_key + str(l)].sum()

            prob_array[i, j] = favorable_events / count_events
            if debug:
                print("volatility - volume ", volatilies_quantiles[i], volume_quantiles[j], "prob:", prob_array[i, j])

    filename = data_dir + "prob_exec.npy"
    with open(filename, 'wb') as f:
        np.save(f, prob_array)


def compute_npy_cost_matrices(FI_dataset : DEEPDataset) -> None:
    out_dir = "data/thresholds_FI/"
    df = FI_dataset.all_df_data.copy().reset_index(drop=True)
    compute_fi_matrices(df, debug=False, data_dir=out_dir)


if __name__ == "__main__":
    # test, load data
    horizon = 20
    out_dir = "data/thresholds_FI/"
    dir_data = "indata/FI-2010/"
    base_lob_dts = DEEPDataset(dir_data, horizon=horizon)
    df_test = base_lob_dts.split_test_data(torch_dataset=False)
    compute_fi_matrices(df_test, debug=False, data_dir=out_dir)


# okay tested -->
"""
>>> df_base.loc[280:284, ["vsell1", "pbuy1", "mid_price", "exec_midprice"]]
      vsell1   pbuy1  mid_price  exec_midprice
280  0.00394  0.2674    0.26745        0.26745
281  0.00394  0.2674    0.26745        0.26740
282  0.00394  0.2673    0.26740        0.26710
283  0.00100  0.2669    0.26710        0.26715
284  0.00215  0.2669    0.26715        0.26705

>>> df_with_executions.loc[280:284, ["vsell1", "pbuy1", "mid_price", "exec_midprice"]]
      vsell1  pbuy1  mid_price  exec_midprice
280  0.00394      0    0.26745        0.26745
281  0.00394      1    0.26745        0.26740
282  0.00394      1    0.26740        0.26710
283  0.00100      0    0.26710        0.26715
284  0.00215      0    0.26715        0.26705


The case on line 281 and 282 are executed, as the price of buy(0,2674 and 0,2673) are >= than the exec_midprice (future mid-price at the next second), 
therefore they will be executed.

-----

The probabilities are also okay, they decrease at increasing of outstanding volume, that make sense:
Volatility mid-price not very helpful here. 

volatility - volume  0.0006135738077969675 0.02201 prob: 0.009419979579065247
volatility - volume  0.0006135738077969675 0.05423 prob: 0.004683163285250772
volatility - volume  0.0006135738077969675 0.16826000000000002 prob: 0.0026864826108893274
volatility - volume  0.0006135738077969675 inf prob: 0.001817807453598847

volatility - volume  0.000913657256536161 0.02201 prob: 0.0073479827926985234
volatility - volume  0.000913657256536161 0.05423 prob: 0.0037434685733848927
volatility - volume  0.000913657256536161 0.16826000000000002 prob: 0.0025308148394007513
volatility - volume  0.000913657256536161 inf prob: 0.001988410742124305

volatility - volume  0.02022076121882971 0.02201 prob: 0.00550221382020795
volatility - volume  0.02022076121882971 0.05423 prob: 0.002931399214264128
volatility - volume  0.02022076121882971 0.16826000000000002 prob: 0.0021384215294665684
volatility - volume  0.02022076121882971 inf prob: 0.0017148571932817601

volatility - volume  inf 0.02201 prob: 0.009507979496239815
volatility - volume  inf 0.05423 prob: 0.005277155739132001
volatility - volume  inf 0.16826000000000002 prob: 0.003525278389805276
volatility - volume  inf inf prob: 0.0026119951760449453

"""