from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from operator import itemgetter
from utilities import utils
import seaborn as sns
import numpy as np
import json
import sys
import os

dir_store = ''
core_num = 1
max_iter = 5000


def cluster():
    """
    Cluster the documents using out of core Mini Batch KMeans. 

    :return: 
    """

    global dir_store, core_num
    config = json.load(open('config.json'))
    dir_store = config['dir_store']
    core_num = config['core_num']

    if len(sys.argv) < 2:
        print('Missing number of clusters')
        exit()
    num_clusters_max = int(sys.argv[1])

    uuids = sorted(os.listdir(dir_store))
    matrix_file = open('data/matrix.txt', 'r')
    data = np.loadtxt(matrix_file)

    # Retrieve base labels
    print('Acquire base labels')
    base_labels = utils.get_base_labels(uuids)
    base_labels = np.asarray(base_labels)

    # num_clusters, silhouette = test_kmeans_clusters(data, base_labels, 2, num_clusters_max)
    test_kmeans_clusters(data, base_labels, num_clusters_max)

    num_clusters = 0
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

    matrix_file2d = open('data/matrix2d.txt', 'r')
    data_red = np.loadtxt(matrix_file2d)

    # color_palette = sns.color_palette('deep', num_clusters)
    # cluster_colors = [color_palette[x] if x >= 0
    #                   else (0.5, 0.5, 0.5)
    #                   for x in computed_labels]
    # cluster_member_colors = [sns.desaturate(x, p) for x, p in
    #                          zip(cluster_colors, hdbs.probabilities_)]
    # plt.scatter(*data_red.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)
    # plt.show()
    #
    # # For visualization
    # reduced_data = PCA(n_components=2).fit_transform(data)

    utils.result_to_visualize(uuids, base_labels, computed_labels, num_clusters)


def test_kmeans_clusters(data, base_labels, num_clusters_max):
    """
    Test several values for the number of clusters 
    
    :param data: data matrix to cluster
    :param base_labels: base labels from AVClass
    :param num_clusters_min: minimum number of clusters to try
    :param num_clusters_max: maximum number of clusters to try
    :return: 
    """

    silhouettes = {}
    mult = 0.1

    while mult * num_clusters_max <= num_clusters_max:
        cur_num_clusters = int(mult * num_clusters_max)
        k_means = KMeans(n_clusters=cur_num_clusters, n_jobs=core_num, max_iter=max_iter, random_state=42)
        computed_labels = k_means.fit_predict(data)

        ars, ami, fm, h, c, sh = utils.evaluate_clustering(base_labels, computed_labels, data=data)
        silhouettes[cur_num_clusters] = sh
        mult += 0.1

    for sh in sorted(silhouettes.items(), key=itemgetter(1), reverse=True):
        print('Clusters {} Silhouette {}'.format(sh[0], sh[1]))


if __name__ == '__main__':
    cluster()
