from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.externals import joblib
from distances import jensen_shannon
from utilities import interaction
from utilities import constants
import hdbscan
import os


# noinspection PyUnusedLocal
def cluster(data, base_labels, config, params):
    """
    Clusters the data matrix using the HDBSCAN algorithm.
    :param data: either a data matrix or a list of document uuids
    :param base_labels: list of labels from a reference clustering
    :param config: configuration dictionary
    :param params: dictionary of parameters for the algorithm
    :return: Clustering labels, trained model and modifiers
    """

    modifier = ''
    core_num = config['core_num']

    min_cluster_size = params.get('min_cluster_size', 60)
    min_sample_param = params.get('min_sample', 15)
    distance = params.get('distance', None)

    if not distance:
        distance = interaction.ask_metric()

    hdbs, clustering_labels, metric = None, None, None

    if distance == 'e':
        modifier = 'euclidean'
        metric = 'euclidean'

    elif distance == 'c':
        modifier = 'cosine'
        metric = 'precomputed'
        data = pairwise_distances(data, metric=modifier)

    elif distance == 'c1':
        modifier = 'cosine1'
        norm_data = normalize(data, norm='l2')
        metric = 'euclidean'

    elif distance == 'j':
        modifier = 'jensen_shannon'
        metric = 'precomputed'

    elif distance == 'j1':
        modifier = 'jensen_shannon1'
        metric = jensen_shannon.jensen_shannon_dist

    hdbs = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_sample_param,
        metric=metric,
        core_dist_n_jobs=core_num
    )

    hdbs.fit(data)
    clustering_labels = hdbs.labels_

    model_file = os.path.join(
        constants.dir_d,
        constants.dir_mod,
        'hdbscan_{}_{}.pkl'.format(modifier, len(data))
    )
    
    joblib.dump(hdbs, model_file)

    return clustering_labels, hdbs, modifier, data, metric
