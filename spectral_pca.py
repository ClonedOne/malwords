from sklearn.cluster import SpectralClustering
from utilities import utils
from sklearn import metrics
import numpy as np
import json
import os

dir_store = ''
num_clusters = 5
core_num = 4


def cluster():
    """
    Cluster the documents using out of core Mini Batch KMeans. 

    :return: 
    """

    global dir_store
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

    spectral = SpectralClustering(n_clusters=num_clusters, n_jobs=core_num)
    computed_labels = spectral.fit_predict(data)

    # Evaluate clustering
    print('Clustering evaluation:', num_clusters)
    print('Adjusted Rand index:', metrics.adjusted_rand_score(base_labels, computed_labels))
    print('Adjusted Mutual Information:', metrics.adjusted_mutual_info_score(base_labels, computed_labels))
    print('Fowlkes-Mallows:', metrics.fowlkes_mallows_score(base_labels, computed_labels))
    print('Homogeneity:', metrics.homogeneity_score(base_labels, computed_labels))
    print('Completeness:', metrics.completeness_score(base_labels, computed_labels))
    print('Silhouette', metrics.silhouette_score(data, computed_labels, metric='euclidean'))
    print('-'*80)

    utils.result_to_visualize(uuids, base_labels, computed_labels)


if __name__ == '__main__':
    cluster()