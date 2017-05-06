from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from utilities import utils
import seaborn as sns
import numpy as np
import hdbscan
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

    matrix_file = open('data/matrix.txt', 'r')
    data = np.loadtxt(matrix_file)

    # Retrieve base labels
    print('Acquire base labels')
    base_labels = utils.get_base_labels(uuids)
    base_labels = np.asarray(base_labels)

    if len(sys.argv) < 2:
        print('Please specify distance metric, either e for euclidean or c for cosine')
        return
    distance_type = sys.argv[1]
    if distance_type == 'e':
        euclidean(data, uuids, base_labels, 30)
    elif distance_type == 'c':
        cosine(data, uuids, base_labels, 30)
    else:
        print('Please specify distance metric, either e for euclidean or c for cosine')


def euclidean(data, uuids, base_labels, min_cluster_size):
    """
    Perform HDBSCAN with euclidean distance
    
    :param data: data to cluster
    :param uuids: list of uuids corresponding to data points
    :param base_labels: reference clustering
    :param min_cluster_size: minimum number of points to generate cluster
    :return: 
    """

    print('Perform clustering')
    hdbs = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, gen_min_span_tree=True)
    hdbs.fit(data)
    computed_labels = hdbs.labels_
    num_clusters = len(set(computed_labels))

    matrix_file2d = open('data/matrix2d.txt', 'r')
    data_red = np.loadtxt(matrix_file2d)

    color_palette = sns.color_palette('deep', num_clusters)
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in hdbs.labels_]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                             zip(cluster_colors, hdbs.probabilities_)]
    plt.scatter(*data_red.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)
    plt.show()

    hdbs.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=80, edge_linewidth=2)
    plt.show()
    hdbs.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
    plt.show()

    utils.evaluate_clustering(base_labels, computed_labels, data=data)

    utils.result_to_visualize(uuids, base_labels, computed_labels, num_clusters)


def cosine(data, uuids, base_labels, min_cluster_size):
    """
    Perform HDBSCAN with cosine distance

    :param data: data to cluster
    :param uuids: list of uuids corresponding to data points
    :param base_labels: reference clustering
    :param min_cluster_size: minimum number of points to generate cluster
    :return: 
    """

    print('Perform clustering')
    distance = pairwise_distances(data, metric='cosine')
    print(distance.shape)

    hdbs = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed')
    hdbs.fit(distance)
    computed_labels = hdbs.labels_
    num_clusters = len(set(computed_labels))

    matrix_file2d = open('data/matrix2d.txt', 'r')
    data_red = np.loadtxt(matrix_file2d)

    color_palette = sns.color_palette('deep', num_clusters)
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in hdbs.labels_]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                             zip(cluster_colors, hdbs.probabilities_)]
    plt.scatter(*data_red.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)
    plt.show()

    utils.evaluate_clustering(base_labels, computed_labels, data=data)

    utils.result_to_visualize(uuids, base_labels, computed_labels, num_clusters)


if __name__ == '__main__':
    cluster()
