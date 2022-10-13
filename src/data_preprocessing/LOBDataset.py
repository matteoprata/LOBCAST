

from torch.utils import data


class Dataset(data.Dataset):
    """ Characterizes a dataset for PyTorch """

    def __init__(self, df_lob, horizon, num_classes=3, goal_col="y", one_hot_encoding=False):
        """Initialization"""
        self.df_lob = df_lob
        self.num_classes = num_classes
        self.horizon = horizon
        self.goal_col = goal_col

        x, y = self.__prepare_observations()

        self.reset_x_y(x, y)

        if one_hot_encoding:
            self.y = F.one_hot(self.y.to(torch.int64), num_classes=self.num_classes)

    def reset_x_y(self, x, y):
        print("resetting x, y")
        self.length = len(x)
        self.unique, self.counts = np.unique(y, return_counts=True)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __prepare_observations(self):
        """ preprocessing data, so to split X and Y """
        relevant_columns = [c for c in self.df_lob.columns if "sell" in c or "buy" in c or "volatility" in c]

        X = self.df_lob[relevant_columns].values
        Y = self.df_lob[self.goal_col].values

        [rows, cols] = X.shape
        dX, dY = np.array(X), np.array(Y)

        # TODO: removes the first HORIZON rows, but why here?
        dataY = dY[self.horizon - 1:]

        # rows - self.horizon + 1: number of rows minus the horizon rows
        # crates the mapping: window size rows -> class
        dataX = np.zeros((rows - self.horizon + 1, self.horizon, cols))

        for i in range(rows - self.horizon + 1):
            observation = dX[self.horizon*i:self.horizon*(i+1), :]
            if observation.shape[0] == self.horizon:
                dataX[i, :, :] = observation  # the i-th observation
        return dataX, dataY

    def __len__(self):
        """ Denotes the total number of samples """
        return self.x.shape[0]

    def __getitem__(self, index):
        """ Generates samples of data """
        return self.x[index], self.y[index]
