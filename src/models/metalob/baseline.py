import numpy as np
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader


class BaselineEnsemble:
    def __init__(self, test_set, f1_scores):
        super().__init__()
        self.test_set = test_set
        self.f1_scores = f1_scores
        self.dataloader = DataLoader(self.test_set, batch_size=1, shuffle=False)

    def run_w_percentage(self):
        preds = []
        for x, y in self.dataloader:
            weighted_x = np.zeros(x.shape[1])
            # weighting the predictions of each model
            for j in range(len(self.f1_scores)):
                weighted_x[3 * j:3 * j + 3] = (x[0, 3 * j:3 * j + 3] * self.f1_scores[j])

            # summing the weighted predictions
            up_trend = np.sum(weighted_x[::3], axis=0)
            stat = np.sum(weighted_x[1::3], axis=0)
            down_trend = np.sum(weighted_x[2::3], axis=0)
            preds.append(np.argmax([up_trend, stat, down_trend]))

        print(classification_report(self.test_set.y, preds, digits=4))

    def run_w_majority(self):
        preds = []
        for x, y in self.dataloader:
            x = x.reshape(-1, 3)
            weighted_x = np.zeros(x.shape[0])
            up_trend = 0
            stat = 0
            down_trend = 0
            # weighting the predictions of each model
            for j in range(len(self.f1_scores)):
                pred = np.argmax(x[j])
                if (pred == 0):
                    up_trend += pred*self.f1_scores[j]
                elif (pred == 1):
                    stat += pred*self.f1_scores[j]
                else:
                    down_trend += pred*self.f1_scores[j]

            preds.append(np.argmax([up_trend, stat, down_trend]))

        print(classification_report(self.test_set.y, preds, digits=4))
