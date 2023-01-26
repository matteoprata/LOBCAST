from sklearn.cluster import KMeans
import numpy as np
import torch


def get_nbof_centers(data_module, k=16, perc=0.4):
    data_v = data_module.train_set.x                                # torch.Size([203700, 100, 40])
    data_v = data_v.view(*data_v.shape[:0], -1, *data_v.shape[2:])  # torch.Size([203700 * 100, 40])
    data_v = data_v.clone().detach().numpy()

    # select perc% random rows in the dataset to compute k-means
    data_v = data_v[np.random.randint(0, data_v.shape[0], int(data_v.shape[0]*perc)), :]

    print("Computing the clusters for a mat", data_v.shape)
    kmeans = KMeans(n_clusters=k, n_init=1).fit(data_v)
    print("Clusters computed")

    clusters = kmeans.cluster_centers_
    t = torch.from_numpy(np.array(clusters))
    return t
