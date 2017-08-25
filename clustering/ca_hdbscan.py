from sklearn.metrics import pairwise_distances
from distances import jensen_shannon
from visualization import vis_plot
from utilities import interaction
from helpers import loader_freqs
from utilities import evaluation
from utilities import constants
from utilities import output
import numpy as np
import hdbscan
import json
import os


def cluster(config, distance, uuids, base_labels, sparse=False):
    """
    Cluster the documents using out of core Mini Batch KMeans. 

    :return: 
    """

    core_num = config['core_num']
    dir_malwords = config['dir_malwords']
    min_cluster_size = 30
    words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))

    distance_type = distance

    if sparse:
        data = loader_freqs.load_freqs(uuids, core_num, len(words), words, dir_malwords, dense=False, ordered=True)
    else:
        matrix_file = interaction.ask_file(constants.msg_data_train)
        data = np.loadtxt(matrix_file)

    if distance_type == 'e':
        hdbs, computed_labels, metric, data = euclidean(data, uuids, base_labels, min_cluster_size, core_num, sparse)
    elif distance_type == 'c':
        hdbs, computed_labels, metric, data = cosine(data, uuids, base_labels, min_cluster_size, core_num, sparse)
    elif distance_type == 'j':
        hdbs, computed_labels, metric, data = js(data, uuids, base_labels, min_cluster_size, core_num, sparse)

    num_clusters = len(set(computed_labels)) - (1 if -1 in computed_labels else 0)
    if num_clusters == 1:
        data = None

    evaluation.evaluate_clustering(base_labels, computed_labels, data=data, metric=metric)
    output.result_to_visualize(uuids, base_labels, computed_labels, num_clusters)
    vis_plot.plot_hdbs_against_2d(hdbs, num_clusters)


def js(data, uuids, base_labels, min_cluster_size, core_num, sparse):
    """
      Perform HDBSCAN with jensen-shannon distance

      :param data: data in distance matrix shape
      :param uuids: list of uuids corresponding to data points
      :param base_labels: reference clustering
      :param min_cluster_size: minimum number of points to generate cluster
      :param core_num: number of available cpu cores
      :param sparse: flag, if set the data matrix is sparse
      :return:
      """

    print('Perform clustering with jensen-shannon distance')

    if not sparse:
        m = 'precomputed'
        hdbs = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=m, core_dist_n_jobs=core_num)
    else:
        m = jensen_shannon.jensen_shannon_dist
        hdbs = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=m, core_dist_n_jobs=core_num)

    hdbs.fit(data)
    computed_labels = hdbs.labels_

    output.out_clustering(dict(zip(uuids, computed_labels.tolist())), 'jensen_shannon', 'hdbscan')

    return hdbs, computed_labels, m, data


def euclidean(data, uuids, base_labels, min_cluster_size, core_num, sparse):
    """
    Perform HDBSCAN with euclidean distance
    
    :param data: data to cluster
    :param uuids: list of uuids corresponding to data points
    :param base_labels: reference clustering
    :param min_cluster_size: minimum number of points to generate cluster
    :param core_num: number of available cpu cores
    :param sparse: flag, if set the data matrix is sparse
    :return: 
    """

    print('Perform clustering with euclidean distance')

    m = 'euclidean'
    hdbs = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=m, gen_min_span_tree=True,
                           core_dist_n_jobs=core_num)
    hdbs.fit(data)
    computed_labels = hdbs.labels_

    output.out_clustering(dict(zip(uuids, computed_labels.tolist())), 'euclidean', 'hdbscan')

    return hdbs, computed_labels, m, data


def cosine(data, uuids, base_labels, min_cluster_size, core_num, sparse):
    """
    Perform HDBSCAN with cosine distance

    :param data: data to cluster
    :param uuids: list of uuids corresponding to data points
    :param base_labels: reference clustering
    :param min_cluster_size: minimum number of points to generate cluster
    :param core_num: number of available cpu cores
    :param sparse: flag, if set the data matrix is sparse
    :return: 
    """

    print('Perform clustering with cosine distance')

    distance = pairwise_distances(data, metric='cosine')
    print(distance.shape)

    m = 'precomputed'
    hdbs = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=m, core_dist_n_jobs=core_num)

    hdbs.fit(distance)
    computed_labels = hdbs.labels_

    output.out_clustering(dict(zip(uuids, computed_labels.tolist())), 'cosine', 'hdbscan')

    return hdbs, computed_labels, m, distance
