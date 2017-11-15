from sklearn.cluster import SpectralClustering
from utilities import constants, interaction
from sklearn.externals import joblib
import numpy as np
import os


# noinspection PyUnusedLocal
def cluster(data, base_labels, config, params):
    """
    Cluster the documents using the Spectral Clustering algorithm.

    :param data: either a data matrix or a list of document uuids
    :param base_labels: list of labels from a reference clustering
    :param config: configuration dictionary
    :param params: dictionary of parameters for the algorithm
    :return: Clustering labels, trained model and modifiers
    """

    core_num = config['core_num']

    seed = params.get('seed', 42)
    delta = params.get('delta', 1)
    modifier = params.get('affinity', 'nearest_neighbors')
    n_neighbors = params.get('n_neighbors', 10)
    isdistance = params.get('isdistance', False)

    num_clusters = params.get('num_clusters', None)
    if not num_clusters:
        num_clusters = interaction.ask_number('clusters')

    # Convert distance matrix to affinity matrix
    if isdistance:
        data = np.exp(- data ** 2 / (2. * delta ** 2))

    spectral = SpectralClustering(
        affinity=modifier,
        random_state=seed,
        n_clusters=num_clusters,
        n_jobs=core_num,
        n_init=20,
        n_neighbors=n_neighbors
    )

    clustering_labels = spectral.fit_predict(data)

    model_file = os.path.join(
        constants.dir_d,
        constants.dir_mod,
        'spectral_{}_{}.pkl'.format('js', len(data))
    )

    joblib.dump(spectral, model_file)

    return clustering_labels, spectral, modifier, data, 'precomputed'
