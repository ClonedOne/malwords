from sklearn.cluster import SpectralClustering
from utilities import utils
import numpy as np
import json
import sys
import os

dir_store = ''
core_num = 1


def cluster():
    """
    Cluster the documents using out of core Mini Batch KMeans. 

    :return: 
    """

    global dir_store, core_num
    config = json.load(open('config.json'))
    dir_store = config['dir_store']
    core_num = config['core_num']
    uuids = sorted(os.listdir(dir_store))

    if not sys.argv[1]:
        print('Missing number of clusters')
        exit()
    num_clusters = sys.argv[1]

    matrix_file = open('data/matrix.txt', 'r')
    data = np.loadtxt(matrix_file)

    # Retrieve base labels
    print('Acquire base labels')
    base_labels = utils.get_base_labels_old(uuids)
    base_labels = np.asarray(base_labels)

    print('Perform clustering')
    spectral = SpectralClustering(n_clusters=num_clusters, n_jobs=core_num)
    computed_labels = spectral.fit_predict(data)

    utils.evaluate_clustering(base_labels, computed_labels, data=data)

    utils.result_to_visualize(uuids, base_labels, computed_labels, num_clusters)


if __name__ == '__main__':
    cluster()