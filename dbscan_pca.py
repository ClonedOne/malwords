from sklearn.cluster import DBSCAN
from utilities import utils
from sklearn import metrics
import numpy as np
import json
import os

dir_store = ''
core_num = 4
max_iter = 1000
num_clusters = 0


def cluster():
    """
    Cluster the documents using out of core Mini Batch KMeans. 

    :return: 
    """

    global dir_store, num_clusters
    config = json.load(open('config.json'))
    dir_store = config['dir_store']
    uuids = sorted(os.listdir(dir_store))

    matrix_file = open('data/matrix.txt', 'r')
    data = np.loadtxt(matrix_file)

    # Retrieve base labels
    base_labels = utils.get_base_labels(uuids)
    base_labels = np.asarray(base_labels)
    print('Base labels')
    print(base_labels)

    dbscan = DBSCAN(n_jobs=core_num, metric='jaccard')
    computed_labels = dbscan.fit_predict(data)
    num_clusters = len(set(computed_labels)) - (1 if -1 in computed_labels else 0)
    print(computed_labels)

    # Evaluate clustering
    print('Clustering evaluation')
    print('Adjusted Rand index:', metrics.adjusted_rand_score(base_labels, computed_labels))
    print('Adjusted Mutual Information:', metrics.adjusted_mutual_info_score(base_labels, computed_labels))
    print('Fowlkes-Mallows:', metrics.fowlkes_mallows_score(base_labels, computed_labels))
    print('Homogeneity:', metrics.homogeneity_score(base_labels, computed_labels))
    print('Completeness:', metrics.completeness_score(base_labels, computed_labels))
    print('Silhouette', metrics.silhouette_score(data, computed_labels, metric='euclidean'))
    print('-'*80)

    utils.result_to_visualize(uuids, base_labels, computed_labels, num_clusters)


if __name__ == '__main__':
    cluster()
