from sklearn.metrics import pairwise_distances
from distances import jensen_shannon
from utilities import interaction
from helpers import loader_freqs
from utilities import evaluation
from utilities import constants
import matplotlib.pyplot as plt
from utilities import output
from utilities import utils
import seaborn as sns
import numpy as np
import hdbscan
import json
import os

core_num = 1


def cluster(config, distance):
    """
    Cluster the documents using out of core Mini Batch KMeans. 

    :return: 
    """

    global core_num
    dir_store = config['dir_store']
    dir_malwords = config['dir_mini']
    core_num = config['core_num']
    min_cluster_size = 30

    distance_type = distance

    words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))
    uuids = sorted(os.listdir(dir_store))

    # Retrieve base labels
    print('Acquiring base labels')
    base_labels_dict = utils.get_base_labels()
    base_labels = np.asarray([base_labels_dict[uuid] for uuid in uuids])

    if distance_type == 'e':
        matrix_file = interaction.ask_file(constants.msg_data)
        data = np.loadtxt(matrix_file)
        euclidean(data, uuids, base_labels, min_cluster_size)

    elif distance_type == 'c':
        matrix_file = interaction.ask_file(constants.msg_data)
        data = np.loadtxt(matrix_file)
        cosine(data, uuids, base_labels, min_cluster_size)

    elif distance_type == 'j':
        js(uuids, base_labels, min_cluster_size, words, dir_malwords)

    else:
        print('Please specify distance metric, either e for euclidean or c for cosine')


def js(uuids, base_labels, min_cluster_size, words, dir_malwords):
    """
      Perform HDBSCAN with jensen-shannon distance

      :param uuids: list of uuids corresponding to data points
      :param base_labels: reference clustering
      :param min_cluster_size: minimum number of points to generate cluster
      :param words: dictionary of valid words with id
      :param dir_malwords: directory containing bag-of-word files
      :return:
      """

    print('Perform clustering with jensen-shannon distance')

    data = loader_freqs.load_freqs(uuids, core_num, len(words), words, dir_malwords, dense=False, ordered=True)
    distance = pairwise_distances(data, metric=jensen_shannon.jensen_shannon_dist, n_jobs=core_num)

    hdbs = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed')
    hdbs.fit(distance)
    computed_labels = hdbs.labels_

    num_clusters = len(set(computed_labels)) - (1 if -1 in computed_labels else 0)

    if num_clusters == 1:
        data = None

    evaluation.evaluate_clustering(base_labels, computed_labels, data=data)

    output.result_to_visualize(uuids, base_labels, computed_labels, num_clusters)

    output.out_clustering(dict(zip(uuids, computed_labels.tolist())), 'jensen_shannon', 'hdbscan')


def euclidean(data, uuids, base_labels, min_cluster_size):
    """
    Perform HDBSCAN with euclidean distance
    
    :param data: data to cluster
    :param uuids: list of uuids corresponding to data points
    :param base_labels: reference clustering
    :param min_cluster_size: minimum number of points to generate cluster
    :return: 
    """

    print('Perform clustering with euclidean distance')
    hdbs = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', gen_min_span_tree=True,
                           match_reference_implementation=True)
    hdbs.fit(data)
    computed_labels = hdbs.labels_

    num_clusters = len(set(computed_labels)) - (1 if -1 in computed_labels else 0)

    if num_clusters == 1:
        data = None

    evaluation.evaluate_clustering(base_labels, computed_labels, data=data)

    output.result_to_visualize(uuids, base_labels, computed_labels, num_clusters)

    output.out_clustering(dict(zip(uuids, computed_labels.tolist())), 'euclidean', 'hdbscan')

    plot_against_2d(hdbs, num_clusters, has_tree=True)


def cosine(data, uuids, base_labels, min_cluster_size):
    """
    Perform HDBSCAN with cosine distance

    :param data: data to cluster
    :param uuids: list of uuids corresponding to data points
    :param base_labels: reference clustering
    :param min_cluster_size: minimum number of points to generate cluster
    :return: 
    """

    print('Perform clustering with cosine distance')
    distance = pairwise_distances(data, metric='cosine')
    print(distance.shape)

    hdbs = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed')
    hdbs.fit(distance)
    computed_labels = hdbs.labels_

    num_clusters = len(set(computed_labels)) - (1 if -1 in computed_labels else 0)

    if num_clusters == 1:
        data = None

    evaluation.evaluate_clustering(base_labels, computed_labels, data=data)

    output.result_to_visualize(uuids, base_labels, computed_labels, num_clusters)

    output.out_clustering(dict(zip(uuids, computed_labels.tolist())), 'cosine', 'hdbscan')

    plot_against_2d(hdbs, num_clusters)


def plot_against_2d(hdbs, num_clusters, has_tree=False):
    """
    Plot the clustering result of HDBSCAN against a 2d projection of data.
    
    :return: 
    """

    matrix_file2d = ""

    # Forces the user to choose the desired number of clusters
    while matrix_file2d == "":
        matrix_file2d = input('Please select the 2d data file or q to quit\n')

        if matrix_file2d == 'q':
            exit()

        try:
            if not os.path.isfile(matrix_file2d):
                raise Exception
        except:
            matrix_file2d = ""
            print('Please select the 2d data file or q to quit\n')

    data_red = np.loadtxt(matrix_file2d)

    color_palette = sns.color_palette('deep', num_clusters)
    cluster_colors = [color_palette[x] if x >= 0
                      else (0.5, 0.5, 0.5)
                      for x in hdbs.labels_]
    cluster_member_colors = [sns.desaturate(x, p) for x, p in
                             zip(cluster_colors, hdbs.probabilities_)]
    plt.scatter(*data_red.T, s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)
    plt.show()

    if has_tree:
        hdbs.minimum_spanning_tree_.plot(edge_cmap='viridis', edge_alpha=0.6, node_size=80, edge_linewidth=2)
        plt.show()
        hdbs.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
        plt.show()
