import collections

import src.config as co
import numpy as np
import tqdm


class FIDataBuilder:
    def __init__(
            self,
            fi_data_dir,
            dataset_type,
            auction=False,
            normalization_type=co.NormalizationType.Z_SCORE,
            horizon=10,
            window=100,
    ):

        assert horizon in (1, 2, 3, 5, 10)

        self.dataset_type = dataset_type
        self.fi_data_dir = fi_data_dir
        self.auction = auction
        self.normalization_type = normalization_type
        self.horizon = horizon
        self.window = window

        # KEY call, generates the dataset
        self.data, self.samples_x, self.samples_y = None, None, None
        self.__prepare_dataset()

    def __read_dataset(self):
        """ Reads the dataset from the FI files. """

        AUCTION = 'Auction' if self.auction else 'NoAuction'
        N = '1.' if self.normalization_type == co.NormalizationType.Z_SCORE else '2.' if self.normalization_type == co.NormalizationType.MINMAX else '3.'
        NORMALIZATION = 'Zscore' if self.normalization_type == co.NormalizationType.Z_SCORE else 'MinMax' if self.normalization_type == co.NormalizationType.MINMAX else 'DecPre'
        DATASET_TYPE = 'Training' if self.dataset_type == co.DatasetType.TRAIN else 'Testing'
        DIR = self.fi_data_dir + \
                 "/{}".format(AUCTION) + \
                 "/{}{}_{}".format(N, AUCTION, NORMALIZATION) + \
                 "/{}_{}_{}".format(AUCTION, NORMALIZATION, DATASET_TYPE)

        NORMALIZATION = 'ZScore' if self.normalization_type == co.NormalizationType.Z_SCORE else 'MinMax' if self.normalization_type == co.NormalizationType.MINMAX else 'DecPre'
        DATASET_TYPE = 'Train' if self.dataset_type == co.DatasetType.TRAIN else 'Test'

        F_EXTENSION = '.txt'


        # if it is training time, we open the 7-days training file
        # if it is testing time, we open the 3 test files
        if self.dataset_type == co.DatasetType.TRAIN:

            F_NAME = DIR + \
                     '/{}_Dst_{}_{}_CF_7'.format(DATASET_TYPE, AUCTION, NORMALIZATION) + \
                     F_EXTENSION

            #print('Opening:', F_NAME)

            out_df = np.loadtxt(F_NAME)

        else:

            F_NAMES = [
                DIR + \
                '/{}_Dst_{}_{}_CF_{}'.format(DATASET_TYPE, AUCTION, NORMALIZATION, i) + \
                F_EXTENSION
                for i in range(7, 10)
            ]

            #print('Opening:', F_NAMES)

            out_df = np.hstack(
                [np.loadtxt(F_NAME) for F_NAME in F_NAMES]
            )

        self.data = out_df

    def __prepareX(self):
        """ we only consider the first 40 features, i.e. the 10 levels of the LOB"""
        self.samples_x = self.data[:40, :].transpose()

    def __prepareY(self):
        """ gets the labels """
        # the last five elements in self.data contain the labels
        # they are based on the possible horizon values [1, 2, 3, 5, 10]
        T = {
            1: -5,
            2: -4,
            3: -3,
            5: -2,
            10: -1
        }

        self.samples_y = self.data[T[self.horizon], :].transpose()
        self.samples_y -= 1

    def __snapshotting(self, do_shuffle=False):
        """ This creates 4 X n_levels X window_size_backward -> prediction. """

        X, Y = [], []
        print("Snapshotting... (__data has", self.data.shape[0], "rows)")
        for st in tqdm.tqdm(range(0, self.samples_x.shape[0] - self.window)):
            x_snap = self.samples_x[st: st + self.window]
            y_snap = self.samples_y[st + self.window]
            X.append(x_snap)
            Y.append(y_snap)

        self.samples_x, self.samples_y = np.asarray(X), np.asarray(Y)

        if do_shuffle:
            index = co.RANDOM_GEN_DATASET.randint(0, self.samples_x.shape[0], size=self.samples_x.shape[0])
            self.samples_x = self.samples_x[index]
            self.samples_y = self.samples_y[index]

    def __prepare_dataset(self):
        """ Crucial call! """

        self.__read_dataset()
        self.__prepareX()
        self.__prepareY()
        self.__snapshotting()

        occurrences = collections.Counter(self.samples_y)
        print("occurrences:", occurrences)

    def get_data(self, first_half_split=1):
        return self.data

    def get_samples_x(self, first_half_split=1):
        return self.samples_x

    def get_samples_y(self, first_half_split=1):
        return self.samples_y
