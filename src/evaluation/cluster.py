import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

def run_kmeans(x, y, k):
    estimator = KMeans(n_clusters=k)

    NMI_list = []
    for i in range(10):
        estimator.fit(x)
        y_pred = estimator.predict(x)
        s = normalized_mutual_info_score(y, y_pred, average_method='arithmetic')
        NMI_list.append(s)

    mean = np.mean(NMI_list)
    std = np.std(NMI_list)
    print('\t[Clustering] NMI: {:.4f} | {:.4f}'.format(mean, std))
    return 