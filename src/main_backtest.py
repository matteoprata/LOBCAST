import json
import sys
from backtesting import Backtest, Strategy
import torch.nn as nn
import constants as cst
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.config import Configuration


def load_predictions(model, stock, seed, period, horizon):

    file_name = "model={}-seed={}-trst=ALL-test={}-data=Lobster-peri={}-bw=1-fw={}-fiw=10.json".format(model,
                                                                                                        seed,
                                                                                                        stock,
                                                                                                        period.name,
                                                                                                        horizon)

    if os.path.exists(cst.DIR_LOBSTER_FINAL_JSONS + file_name):
        with open(cst.DIR_LOBSTER_FINAL_JSONS + file_name, "r") as f:
            d = json.loads(f.read())

            logits_str = d['LOGITS']
            logits = np.array(json.loads(logits_str))

            if (model == cst.Models.DEEPLOBATT):
                horizons = [horizon.value for horizon in cst.FI_Horizons]
                h = horizons.index(horizon)
                logits = logits[:, :, h]

    else:
        print("problem with file", cst.DIR_LOBSTER_FINAL_JSONS + file_name)
        exit()

    preds = np.argmax(logits, axis=1)

    return preds


def load_OHLC(stock, period):
    logits = list()
    file_name = "OHLC_{}.data".format(stock)

    if os.path.exists(cst.DIR_OHLC + file_name):
        df = pd.read_pickle(cst.DIR_OHLC + file_name)

    else:
        print("problem with file", cst.DIR_OHLC + file_name)
        exit()

    return df

class DLstrategy1(Strategy):
    def init(self):
        print("starting")

    def next(self):
        # Proceed only with out-of-sample data. Prepare some variables
        high, low, close = self.data.High[-1], self.data.Low[-1], self.data.Close[-1]
        pred = self.data.Preds[-1]
        # print(self.position)

        # we predict the price will go up
        if int(pred) == 2:

            # if we are short and the price is going up, we close the position and go long
            if self.position.is_short:
                self.position.close()
                self.buy(size=1)

            # if we are not in a position, we go long
            elif self.position.is_long == False and self.position.is_short == False:
                self.buy(size=1)

        # we predict the price will go down
        elif int(pred) == 0:

            # if we are long and the price is going down, we close the position and go short
            if self.position.is_long:
                self.position.close()
                self.sell(size=1)

            # if we are not in a position, we go short
            elif self.position.is_long == False and self.position.is_short == False:
                self.sell(size=1)


class DLstrategy2(Strategy):

    def init(self):
        print("starting")

    def next(self):

        # Proceed only with out-of-sample data. Prepare some variables
        high, low, close = self.data.High[-1], self.data.Low[-1], self.data.Close[-1]
        pred = self.data.Preds[-1]
        #print(self.position)

        # we predict the price will go up
        if int(pred) == 2:

            #if we are not already long we go long
            if self.position.is_long == False and self.position.is_short == False:
                self.buy(size=1)

        # we predict the price will go down
        elif int(pred) == 0:

            # if we are long and the price is going down, we sell
            if self.position.is_long:
                self.position.close()



def run_backtest(cf):

    horizons = [horizon.value for horizon in cst.FI_Horizons]
    Models = [model.name for model in cst.Models if model.name != "METALOB" and model.name != "MAJORITY"]
    Stocks = [stock.name for stock in cst.Stocks if stock.name != "ALL" and stock.name != "FI"]

    for horizon in horizons:
        Returns = np.zeros((len(cst.Models)-2, len(cst.Stocks)-2))
        for i, model in enumerate(Models):
            for j, stock in enumerate(Stocks):

                print(f"Backtest ready for stock {stock} and model {model}")

                # load OHLC and predictions
                OHLC = load_OHLC(stock, cst.Periods.JULY2021)

                preds = load_predictions(model, stock, cf.SEED, cst.Periods.JULY2021, horizon)

                if (stock == "LSTR"):
                    diff_lstr = len(OHLC) - preds.shape[0] - horizon + 1
                    OHLC = OHLC.iloc[diff_lstr:]

                n_instances = len(OHLC)
                '''
                print("len OHLC: " + str(n_instances))
                print("numero di predizioni: " + str(preds.shape[0]))
                print("differenza tra len di OHLC e numero di predizioni: " + str(n_instances - preds.shape[0]))
                print()
                '''
                # we remove the last n_instances - preds.shape[0] rows from the OHLC dataframe
                if (preds.shape[0] < n_instances):
                    diff = n_instances - preds.shape[0]
                    OHLC = OHLC.iloc[:-diff]

                # return the values to the original scale
                OHLC = OHLC.div(10000)

                # We add the predictions to the OHLC dataframe and rename the columns as the library requires
                OHLC['Preds'] = preds
                OHLC.rename(columns={'low': 'Low'}, inplace=True)
                OHLC.rename(columns={'high': 'High'}, inplace=True)
                OHLC.rename(columns={'open': 'Open'}, inplace=True)
                OHLC.rename(columns={'close': 'Close'}, inplace=True)
                OHLC = OHLC[['Open', 'High', 'Low', 'Close', 'Preds']]

                # we run the backtest and print the results
                bt = Backtest(OHLC, DLstrategy2, cash=10000, commission=0, margin=1, trade_on_close=False)
                stats = bt.run()
                #print(stats)
                #bt.plot()
                Returns[i, j] = stats[6]

        table_plot(Returns, horizon)
        #box_plot(Returns, horizon)
        #np.save("Returns_{}".format(horizon), Returns)

def table_plot(Returns, horizon):

    #Returns = np.random.rand(len(cst.Models)-2, len(cst.Stocks)-2)

    models = [model.name for model in cst.Models if model.name != "METALOB" and model.name != "MAJORITY"]
    stocks = [stock.name for stock in cst.Stocks if stock.name != "ALL" and stock.name != "FI"]
    fig, ax = plt.subplots(figsize=(30, 12))
    ax.matshow(Returns, cmap=plt.cm.Greens)

    # set the ticks to be at the middle of each cell
    ax.set_xticks(np.arange(len(stocks)), minor=False)
    ax.set_yticks(np.arange(len(models)), minor=False)

    ax.set_xticklabels(stocks)
    ax.set_yticklabels(models)

    for i in range(len(models)):
        for j in range(len(stocks)):
            ax.text(j, i, round(Returns[i, j], 1), ha="center", va="center")

    plt.title("Returns for horizon = {}".format(horizon))
    #plt.savefig("Returns for horizon = {}.pdf".format(horizon))
    plt.show()


def box_plot(Returns, horizon):

    #Returns = np.random.rand(len(cst.Models)-2, len(cst.Stocks)-2)
    Returns = Returns.T
    models = [model.name for model in cst.Models if model.name != "METALOB" and model.name != "MAJORITY"]
    df = pd.DataFrame(Returns, columns=models)
    plt.rcParams['figure.figsize'] = [13, 9]
    plt.boxplot(df)
    plt.xticks(np.arange(len(models)), models, rotation=90)
    plt.title("Boxplot of the returns for horizon " + str(horizon))
    #plt.savefig("Boxplot_{}.pdf".format(horizon))
    plt.show()


if __name__ == "__main__":
    cf = Configuration()
    run_backtest(cf)