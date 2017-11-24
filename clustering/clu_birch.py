from sklearn.externals import joblib
from sklearn.cluster import Birch
from helpers import loader_tfidf
from utilities import constants
import numpy as np
import random
import os


# noinspection PyUnusedLocal
def cluster(data, base_labels, config, params):
    """
    Clusters the data matrix using the Birch algorithm.
    :param data: either a data matrix or a list of document uuids
    :param base_labels: list of labels from a reference clustering
    :param config: configuration dictionary
    :param params: dictionary of parameters for the algorithm
    :return: Clustering labels, trained model and modifiers
    """

    if isinstance(data, list):
        modifier = 'mini'
    else:
        modifier = 'full'

    seed = params.get('seed', 42)
    random.seed(seed)

    num_clusters = 0

    clustering_labels, birch = test_clusters(config, data, modifier, params)

    model_file = os.path.join(
        constants.dir_d,
        constants.dir_mod,
        'birch_{}_{}_{}.pkl'.format(modifier, num_clusters, len(data))
    )
    joblib.dump(birch, model_file)

    if modifier == 'mini':
        data = None

    return clustering_labels, birch, modifier, data, 'euclidean'


def test_clusters(config, data, modifier, params):
    """
    Perform Birch clustering with or without the 3rd Phase external clustering algorithm (ref. paper).

    :param config: global configuration dictionary
    :param data: either a data matrix or a list of document uuids
    :param modifier: 'mini' for mini-batches, 'full' default
    :param params: dictionary of parameters for the algorithm
    :return: tested models
    """

    cur_num_clusters = params.get('num_clusters', None)
    threshold = params.get('threshold', 1.7)
    branching_factor = params.get('branching_factor', 50)

    mini_batch_size = config['batch_size']

    if modifier == 'mini':
        rows = len(data)
        rand_uuids = random.sample(data, len(data))

        birch = Birch(
            n_clusters=cur_num_clusters,
            threshold=threshold,
            branching_factor=branching_factor
        )

        train(config, birch, rows, rand_uuids, mini_batch_size)

        clustering_labels = apply(config, birch, rows, data, mini_batch_size)

    else:  # modifier = 'full'
        birch = Birch(
            n_clusters=cur_num_clusters,
            threshold=threshold,
            branching_factor=branching_factor
        )

        clustering_labels = birch.fit_predict(data)

    return clustering_labels, birch


def train(config, birch, rows, rand_uuids, mini_batch_size):
    """
    Train the MiniBatchKMeans clustering algorithm with fixed size data batches taken random from the input data-set.

    :param config: global configuration dictionary
    :param mini_batch_size: size of each mini batch
    :param birch: Birch object
    :param rows: number of rows of the data set matrix
    :param rand_uuids: list of documents in randomized order
    :return:
    """

    clustered = 0

    # Divide the docuements in mini batches of fixed size and train the KMeans
    while clustered < rows:
        print('Processing documents from {} to {}'.format(clustered, (clustered + mini_batch_size - 1)))

        data = loader_tfidf.load_tfidf(
            config,
            rand_uuids[clustered:][:mini_batch_size],
            dense=True,
            ordered=False
        )

        clustered += mini_batch_size

        birch.partial_fit(data)


def apply(config, birch, rows, uuids, mini_batch_size):
    """
    Apply the MiniBatchKMeans clustering algorithm with fixed size data batches taken in order from the input data set.

    :param config: global configuration dictionary
    :param mini_batch_size: size of each mini batch
    :param birch: Birch object
    :param rows: number of rows of the data set matrix
    :param uuids: list of documents in order
    :return: labels computed by KMeans over the data set
    """

    clustered = 0
    computed_labels = np.array([])

    # Divide the documents in mini batches of fixed size and apply KMeans on them
    while clustered < rows:
        print('Predicting documents from {} to {}'.format(clustered, (clustered + mini_batch_size - 1)))

        data = loader_tfidf.load_tfidf(
            config,
            uuids[clustered:][:mini_batch_size],
            dense=True,
            ordered=True
        )

        clustered += mini_batch_size

        batch_computed_labels = birch.predict(data)

        computed_labels = np.append(computed_labels, batch_computed_labels)

    return computed_labels
