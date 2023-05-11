
import os
import sys
import torch

# preamble needed for cluster
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)

from src.main_single import *
kset, mset = cst.FI_Horizons, cst.Models
out_data = "final_data/LOBSTER-TESTS/all-mid-plot/"

from src.data_preprocessing.LOB.LOBDataset import LOBDataset

import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from enum import Enum
from sklearn.cluster import MeanShift, KMeans
import numpy as np
import pickle

import src.utils.lob_util as lbu
import src.config as co
import src.constants as cst

np.random.seed(0)
plt.rcParams["figure.figsize"] = (16,9)

print(os.getcwd())

PATH_NAME = "data/LOBSTER_6/unzipped/"
STOCKS_FNAME = ["_data_dwn_48_332__{}_2021-07-01_2021-08-06_10", "_data_dwn_48_332__{}_2022-02-01_2022-02-28_10"]
STOCKS       = ["SOFI", "NFLX", "CSCO", "WING", "SHLS", "LSTR"]


class Period(Enum):
    JULY = 0
    FEB = 1


class Granularity(Enum):
    SEC = 1
    MIN = 60
    FIVE_MIN = int(60 * 5)
    TEN_MIN = int(60*10)
    HALF_HOUR = int(60*30)
    HOUR = 60*60
    DAY = int(60*60*6.5)
    TWO_DAYS = int(60*60*6.5*2)
    THREE_DAYS = int(60*60*6.5*3)
    FIVE_DAYS = int(60*60*6.5*5)


MID_PRICES = {}

for scen in Period:
    daily_midprice = pd.DataFrame()
    RETURNS_FNAME = "data/LOBSTER_6/price_return_df_5_scen{}.dat".format(scen.value)

    if not os.path.exists(RETURNS_FNAME):
        for i, i_stock in enumerate(STOCKS):
            print("Processing stock", i, i_stock)

            stock_path = PATH_NAME + STOCKS_FNAME[scen.value].format(i_stock)

            tra_days = os.listdir(stock_path)
            tra_days.sort()

            out_df = lbu.from_folder_to_unique_df(
                stock_path,
                level=10,
                granularity=cst.Granularity.Sec1,
                boundaries_purge=0
            )
            out_df = out_df.fillna(method="ffill")
            daily_midprice[i_stock] = (out_df.loc[:, "psell1"] + out_df.loc[:, "pbuy1"]) / (2 * 10000)

            print("HAS CORRECT SHAPE?", out_df.shape[0], out_df.shape[0] == 60 * 60 * 6.5 * 25)
            out_df = None

        MID_PRICES[scen] = daily_midprice

        with open(RETURNS_FNAME, "wb") as f:
            pickle.dump(daily_midprice, f)
    else:
        print("Loading pickle...")
        with open(RETURNS_FNAME, "rb") as f:
            daily_midprice = pickle.load(f)

        MID_PRICES[scen] = daily_midprice


df_jul = MID_PRICES[Period.JULY]
df_jul.drop(df_jul.index[(np.where((df_jul.index > '2021-07-16')))[0]], inplace=True)
MID_PRICES[Period.JULY] = df_jul

df_feb = MID_PRICES[Period.FEB]
df_feb.drop(df_feb.index[(np.where((df_feb.index > '2022-02-16')))[0]], inplace=True)
MID_PRICES[Period.FEB] = df_feb


# def get_trading_day(df, day, gran=Granularity.HOUR):
#     return df[gran.value*day:gran.value*(day+1)]
#
# i = 0
# while True:
#     a = get_trading_day(MID_PRICES[Period.FEB], day=i, gran=Granularity.DAY)
#     print(a.index[0], a.index[-1])
#     i += 1
#
# print(df_jul)
# print(df_feb)
#
# exit()

def eval_norm_mid(daily_midprice, gran=1, smooth=1):
    midprice_ret = daily_midprice.iloc[::gran.value, :]
    midprice_ret = midprice_ret / midprice_ret.iloc[0, :]
    midprice_ret = midprice_ret.rolling(smooth).mean()
    return midprice_ret


def eval_returns(daily_midprice, gran=Granularity.HOUR):
    norm = eval_norm_mid(daily_midprice, gran=Granularity.HOUR.SEC).iloc[::gran.value, :]
    rets = (norm - norm.shift(-1)) / norm.shift(-1)
    rets = rets.dropna()
    return rets


def increment_volatiltuy(df_jul, df_mar, gran=Granularity.HOUR):
    period_stats = pd.DataFrame()
    ret_jul = eval_returns(df_jul, gran)
    ret_mar = eval_returns(df_mar, gran)

    period_stats["vol_jul"] = ret_jul.describe().loc['std']
    period_stats["vol_mar"] = ret_mar.describe().loc['std']
    period_stats["perc_vol_%"] = ((period_stats["vol_mar"] - period_stats["vol_jul"]) / period_stats["vol_jul"])*100  # how much more volatile was MARCH wrt to JULY
    stats = period_stats.describe()

    me = round(stats.loc['mean', 'perc_vol_%'])
    st = round(stats.loc['std', 'perc_vol_%'])
    # print(me, st)
    print("WAR Feb 2022 period is {}+/-{}% more volatile ({}) than the July 2021 period.".format(me, st, gran))


increment_volatiltuy(MID_PRICES[Period.JULY], MID_PRICES[Period.FEB], gran=Granularity.HOUR)
increment_volatiltuy(MID_PRICES[Period.JULY], MID_PRICES[Period.FEB], gran=Granularity.MIN)
increment_volatiltuy(MID_PRICES[Period.JULY], MID_PRICES[Period.FEB], gran=Granularity.DAY)
increment_volatiltuy(MID_PRICES[Period.JULY], MID_PRICES[Period.FEB], gran=Granularity.TWO_DAYS)
increment_volatiltuy(MID_PRICES[Period.JULY], MID_PRICES[Period.FEB], gran=Granularity.THREE_DAYS)

def setup_plotting_env():
    plt.rcParams["figure.figsize"] = [16, 9]
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.size"] = 20
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["xtick.labelsize"] = 16
    plt.rcParams["ytick.labelsize"] = 16
    plt.rcParams["font.family"] = "serif"


def plot_mids(mids_df, period, save_path):
    setup_plotting_env()

    mids_df.index = [str(ind) for ind in mids_df.index]
    x_axis = [x.split(" ")[0] for x in list(mids_df.index)]
    fig, ax = plt.subplots(figsize=(13, 8))
    ax = mids_df.plot(ax = ax)

    # lines = {
    #     Period.JULY : ('2021-07-08', '2021-07-12', '2021-08-14'),
    #     Period.FEB : ('2022-02-07', '2022-02-11', '2022-02-15')
    # }
    # ax.axvline(x=x_axis.index(lines[period][0]), color='black')  # 13 mezz'ore in 6.5 ore
    # ax.axvline(x=x_axis.index(lines[period][1]), color='black')
    # ax.axvline(x=x_axis.index(lines[period][2]), color='black')
    #
    EVERY = 200
    xx = [0, int(len(mids_df.index)/2), len(mids_df.index)-1]
    plt.xticks(xx, np.array(x_axis)[xx])

    plt.ylabel("Midprice (normalized)")
    perios = "February 2022" if period == Period.FEB else "July 2021"
    plt.title("Midprice on {}".format(perios))

    plt.xlabel("date")
    plt.ylim([.78, 1.085])
    ax.legend(loc='lower left', fontsize=20)
    plt.tight_layout()
    fig.savefig(save_path + 'scenario-{}.pdf'.format(period))


OUT = ""
plot_mids(eval_norm_mid(MID_PRICES[Period.FEB], gran=Granularity.FIVE_MIN), Period.FEB, save_path=OUT)
plot_mids(eval_norm_mid(MID_PRICES[Period.JULY], gran=Granularity.FIVE_MIN), Period.JULY, save_path=OUT)


