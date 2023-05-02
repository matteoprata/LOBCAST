import pickle
import numpy as np
import pandas as pd
import tqdm

STOCK_NAMES = ['CSCO', 'LSTR', 'NFLX', 'SHLS', 'SOFI', 'WING']


for stock_name in STOCK_NAMES:

    with open(f'data/pickles/JULY2021_{stock_name}_2021-07-13_2021-07-15_test_Granularity.Events1.pickle', 'rb') as f:
        df = pickle.load(f)

    df['midprice'] = (df['pbuy1'] + df['psell1']) / 2

    splitted = df.groupby(df.index.day)

    l = list()
    for _, df_day in splitted:
        l.append(df_day['midprice'].values)

    l_open, l_close, l_high, l_low = list(), list(), list(), list()
    for day_arr in l:
        day_arr = day_arr[:-(len(day_arr) % 10)]
        day_arr = day_arr.reshape((-1, 10))

        open_ = day_arr[:, 0]
        close = day_arr[:, -1]
        high = day_arr.max(1)
        low = day_arr.min(1)

        l_open.append(open_)
        l_close.append(close)
        l_high.append(high)
        l_low.append(low)

    open_ = np.concatenate(l_open)
    close = np.concatenate(l_close)
    high = np.concatenate(l_high)
    low = np.concatenate(l_low)

    outdf = pd.DataFrame()
    outdf['open'] = open_
    outdf['close'] = close
    outdf['high'] = high
    outdf['low'] = low

    outdf.to_csv(f'data/pickles/ohlc/OHLC_JULY2021_{stock_name}_2021-07-13_2021-07-15_test_Granularity.Events10.csv')

    print(f'{stock_name}: {outdf.shape}')
