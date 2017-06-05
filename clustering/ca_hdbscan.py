from sklearn.metrics import pairwise_distances
from visualization import vis_plot
from utilities import interaction
from utilities import evaluation
from utilities import constants
from utilities import output
from utilities import utils
import numpy as np
import hdbscan
import os

core_num = 1


def cluster(config, distance, uuids):
    """
    Cluster the documents using out of core Mini Batch KMeans. 

    :return: 
    """

    global core_num
    core_num = config['core_num']
    min_cluster_size = 30

    distance_type = distance

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
        data = np.loadtxt(os.path.join(constants.dir_d, constants.file_js))
        js(data, uuids, base_labels, min_cluster_size)

    else:
        print('Please specify distance metric, either e for euclidean or c for cosine')


def js(data, uuids, base_labels, min_cluster_size):
    """
      Perform HDBSCAN with jensen-shannon distance

      :param data: data in distance matrix shape
      :param uuids: list of uuids corresponding to data points
      :param base_labels: reference clustering
      :param min_cluster_size: minimum number of points to generate cluster
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
