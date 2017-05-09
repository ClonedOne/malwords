from sklearn.cluster import KMeans
from operator import itemgetter
from utilities import utils
import numpy as np
import json
import sys
import os

dir_store = ''
core_num = 1
max_iter = 5000


def cluster():
    """
    Cluster the documents using KMeans algorithm. 

    :return: 
    """

    global dir_store, core_num
    config = json.load(open('config.json'))
    dir_store = config['dir_store']
    core_num = config['core_num']

    if len(sys.argv) < 3:
        print('Please provide the data matrix file and the desired maximum number of clusters')
        exit()

    matrix_file = sys.argv[1]
    num_clusters_max = int(sys.argv[2])

    uuids = sorted(os.listdir(dir_store))
    data = np.loadtxt(matrix_file)

    # Retrieve base labels
    print('Acquiring base labels')
    base_labels_dict = utils.get_base_labels()
    base_labels = np.asarray([base_labels_dict[uuid] for uuid in uuids])

    test_kmeans_clusters(data, base_labels, num_clusters_max)
    num_clusters = 0

    # Forces the user to choose the desired number of clusters
    while num_clusters == 0:
        num_clusters = input('Please select the number of clusters\n')
        try:
            num_clusters = int(num_clusters)
            if num_clusters > num_clusters_max or num_clusters < 2:
                raise Exception
        except:
            num_clusters = 0
            print('Please insert a valid number of clusters\n')

    k_means = KMeans(n_clusters=num_clusters, n_jobs=core_num, max_iter=max_iter, random_state=42)
    computed_labels = k_means.fit_predict(data)

    utils.evaluate_clustering(base_labels, computed_labels, data=data)

    utils.result_to_visualize(uuids, base_labels, computed_labels, num_clusters)


def test_kmeans_clusters(data, base_labels, num_clusters_max):
    """
    Test several values for the number of clusters. 
    Tries 10 different clusterings increasing by 10% the current clusters number up to the maximum specified value.
    
    :param data: data matrix to cluster
    :param base_labels: base labels from AVClass
    :param num_clusters_max: maximum number of clusters to try
    :return: 
    """

    silhouettes = {}

    for i in range(1, 11):
        mult = 0.1 * i
        cur_num_clusters = int(mult * num_clusters_max)

        if cur_num_clusters >= 2:
            k_means = KMeans(n_clusters=cur_num_clusters, n_jobs=core_num, max_iter=max_iter, random_state=42)
            computed_labels = k_means.fit_predict(data)

            ars, ami, fm, h, c, sh = utils.evaluate_clustering(base_labels, computed_labels, data=data)
            silhouettes[cur_num_clusters] = sh

    print('-' * 80)
    for sh in sorted(silhouettes.items(), key=itemgetter(1), reverse=True):
        print('Clusters {} Silhouette {}'.format(sh[0], sh[1]))


if __name__ == '__main__':
    cluster()
