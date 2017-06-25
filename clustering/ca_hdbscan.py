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


def cluster(config, distance, uuids, base_labels):
    """
    Cluster the documents using out of core Mini Batch KMeans. 

    :return: 
    """

    core_num = config['core_num']
    dir_malwords = config['dir_malwords']
    min_cluster_size = 30
    words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))

    distance_type = distance

    if distance_type == 'e':
        matrix_file = interaction.ask_file(constants.msg_data_train)
        data = np.loadtxt(matrix_file)
        euclidean(data, uuids, base_labels, min_cluster_size, core_num)

    elif distance_type == 'c':
        matrix_file = interaction.ask_file(constants.msg_data_train)
        data = np.loadtxt(matrix_file)
        cosine(data, uuids, base_labels, min_cluster_size, core_num)

    elif distance_type == 'j':
        data = np.loadtxt(os.path.join(constants.dir_d, constants.file_js))
        # data = loader_freqs.load_freqs(uuids, core_num, len(words), words, dir_malwords, dense=False, ordered=True)
        js(data, uuids, base_labels, min_cluster_size, core_num)
        # js2(data, uuids, base_labels, min_cluster_size, core_num)

    else:
        print('Please specify distance metric, either e for euclidean, c for cosine, j for jensen-shannon')


def js(data, uuids, base_labels, min_cluster_size, core_num):
    """
      Perform HDBSCAN with jensen-shannon distance

      :param data: data in distance matrix shape
      :param uuids: list of uuids corresponding to data points
      :param base_labels: reference clustering
      :param min_cluster_size: minimum number of points to generate cluster
      :param core_num: number of available cpu cores
      :return:
      """

    print('Perform clustering with jensen-shannon distance')

    hdbs = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed', core_dist_n_jobs=core_num,
                           match_reference_implementation=True)
    hdbs.fit(data)
    computed_labels = hdbs.labels_

    num_clusters = len(set(computed_labels)) - (1 if -1 in computed_labels else 0)

    if num_clusters == 1:
        data = None

    evaluation.evaluate_clustering(base_labels, computed_labels, data=data, metric='precomputed')

    output.result_to_visualize(uuids, base_labels, computed_labels, num_clusters)

    output.out_clustering(dict(zip(uuids, computed_labels.tolist())), 'jensen_shannon', 'hdbscan')


def js2(data, uuids, base_labels, min_cluster_size, core_num):
    """
      Perform HDBSCAN with jensen-shannon distance

      :param data: data in distance matrix shape
      :param uuids: list of uuids corresponding to data points
      :param base_labels: reference clustering
      :param min_cluster_size: minimum number of points to generate cluster
      :param core_num: number of available cpu cores
      :return:
      """

    print('Perform clustering with jensen-shannon distance')

    metric_js = jensen_shannon.jensen_shannon_dist

    hdbs = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric_js, core_dist_n_jobs=core_num)
    hdbs.fit(data)
    computed_labels = hdbs.labels_

    num_clusters = len(set(computed_labels)) - (1 if -1 in computed_labels else 0)

    if num_clusters == 1:
        data = None

    evaluation.evaluate_clustering(base_labels, computed_labels, data=data, metric='precomputed')

    output.result_to_visualize(uuids, base_labels, computed_labels, num_clusters)

    output.out_clustering(dict(zip(uuids, computed_labels.tolist())), 'jensen_shannon', 'hdbscan')


def euclidean(data, uuids, base_labels, min_cluster_size, core_num):
    """
    Perform HDBSCAN with euclidean distance
    
    :param data: data to cluster
    :param uuids: list of uuids corresponding to data points
    :param base_labels: reference clustering
    :param min_cluster_size: minimum number of points to generate cluster
    :param core_num: number of available cpu cores
    :return: 
    """

    print('Perform clustering with euclidean distance')
    hdbs = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', gen_min_span_tree=True,
                           match_reference_implementation=True, core_dist_n_jobs=core_num)
    hdbs.fit(data)
    computed_labels = hdbs.labels_

    num_clusters = len(set(computed_labels)) - (1 if -1 in computed_labels else 0)

    if num_clusters == 1:
        data = None

    evaluation.evaluate_clustering(base_labels, computed_labels, data=data)

    output.result_to_visualize(uuids, base_labels, computed_labels, num_clusters)

    output.out_clustering(dict(zip(uuids, computed_labels.tolist())), 'euclidean', 'hdbscan')

    vis_plot.plot_hdbs_against_2d(hdbs, num_clusters, has_tree=True)


def cosine(data, uuids, base_labels, min_cluster_size, core_num):
    """
    Perform HDBSCAN with cosine distance

    :param data: data to cluster
    :param uuids: list of uuids corresponding to data points
    :param base_labels: reference clustering
    :param min_cluster_size: minimum number of points to generate cluster
    :param core_num: number of available cpu cores
    :return: 
    """

    print('Perform clustering with cosine distance')
    distance = pairwise_distances(data, metric='cosine')
    print(distance.shape)

    hdbs = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='precomputed', core_dist_n_jobs=core_num,
                           match_reference_implementation=True)
    hdbs.fit(distance)
    computed_labels = hdbs.labels_

    num_clusters = len(set(computed_labels)) - (1 if -1 in computed_labels else 0)

    if num_clusters == 1:
        data = None

    evaluation.evaluate_clustering(base_labels, computed_labels, data=data)

    output.result_to_visualize(uuids, base_labels, computed_labels, num_clusters)

    output.out_clustering(dict(zip(uuids, computed_labels.tolist())), 'cosine', 'hdbscan')

    vis_plot.plot_hdbs_against_2d(hdbs, num_clusters)
