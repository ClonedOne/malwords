from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.externals import joblib
from utilities import interaction
from utilities import evaluation
from helpers import loader_tfidf
from utilities import constants
import numpy as np
import random
import os


def cluster(data, base_labels, config, params):
    """
    Clusters the data matrix using the K-Means algorithm.
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

    tested_models = test_kmeans_clusters(config, data, base_labels, modifier, params)

    # Forces the user to choose one of the tested models
    num_clusters = 0
    while num_clusters not in tested_models:
        num_clusters = interaction.ask_number('clusters')

    clustering_labels, k_means = tested_models[num_clusters]

    model_file = os.path.join(
        constants.dir_d,
        constants.dir_mod,
        'kmeans_{}_{}_{}.pkl'.format(modifier, num_clusters, len(data))
    )
    joblib.dump(k_means, model_file)

    if modifier == 'mini':
        data = None

    return clustering_labels, k_means, modifier, data, 'euclidean'


def test_kmeans_clusters(config, data, base_labels, modifier, params):
    """
    Test several values for the number of clusters.
    Tries 10 different clusterings increasing by 10% of the maximum specified value.

    :param config: global configuration dictionary
    :param data: either a data matrix or a list of document uuids
    :param base_labels: base labels from AVClass
    :param modifier: 'mini' for mini-batches, 'full' default
    :param params: dictionary of parameters for the algorithm
    :return: tested models
    """

    tested_models = {}

    max_iter = params.get('num_iters', 3000)
    num_clusters_max = params.get('num_clusters', len(set(base_labels)))
    seed = params.get('seed', 42)

    core_num = config['core_num']
    mini_batch_size = config['batch_size']

    for mult in np.arange(0.1, 1.1, 0.1):
        cur_num_clusters = int(mult * num_clusters_max)

        if cur_num_clusters < 2:
            continue

        if modifier == 'mini':
            rows = len(data)
            rand_uuids = random.sample(data, len(data))

            k_means = MiniBatchKMeans(
                n_clusters=cur_num_clusters,
                batch_size=mini_batch_size,
                max_iter=max_iter,
                random_state=seed
            )

            train_k_means(config, k_means, rows, rand_uuids, mini_batch_size)

            clustering_labels = apply_k_means(config, k_means, rows, data, mini_batch_size)

            data_eval = None

        else:  # modifier = 'full'
            k_means = KMeans(
                n_clusters=cur_num_clusters,
                n_jobs=core_num,
                max_iter=max_iter,
                random_state=seed
            )

            clustering_labels = k_means.fit_predict(data)

            data_eval = data

        evaluation.evaluate_clustering(base_labels, clustering_labels, data_eval)

        tested_models[cur_num_clusters] = (clustering_labels, k_means)

    return tested_models


def train_k_means(config, k_means, rows, rand_uuids, mini_batch_size):
    """
    Train the MiniBatchKMeans clustering algorithm with fixed size data batches taken random from the input data-set.

    :param config:global configuration dictionary
    :param mini_batch_size: size of each mini batch
    :param k_means: MiniBatchKMeans object
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

        k_means.partial_fit(data)


def apply_k_means(config, k_means, rows, uuids, mini_batch_size):
    """
    Apply the MiniBatchKMeans clustering algorithm with fixed size data batches taken in order from the input data set.

    :param config:global configuration dictionary
    :param mini_batch_size: size of each mini batch
    :param k_means: MiniBatchKMeans object
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

        batch_computed_labels = k_means.predict(data)

        computed_labels = np.append(computed_labels, batch_computed_labels)

    return computed_labels
