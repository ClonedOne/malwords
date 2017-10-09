from sklearn.cluster import MiniBatchKMeans
from sklearn.externals import joblib
from utilities import interaction
from helpers import loader_tfidf
from utilities import evaluation
from utilities import constants
from utilities import output
from scipy.sparse import *
import numpy as np
import random
import os


def cluster(config, clusters, uuids, base_labels):
    """
    Cluster the documents using out of core Mini Batch KMeans. 
    
    :param config: global configuration dictionary
    :param clusters: maximum number of clusters
    :param uuids: list of sekected
    :param base_labels: list of selected base labels
    :return:
    """

    mini_batch_size = config['batch_size']
    rand_uuids = random.sample(uuids, len(uuids))
    rows = len(uuids)

    test_kmeans_clusters(config, base_labels, clusters, mini_batch_size, rows, uuids, rand_uuids)

    # Forces the user to choose the desired number of clusters
    num_clusters = interaction.ask_clusters(clusters)

    k_means = MiniBatchKMeans(n_clusters=num_clusters, batch_size=mini_batch_size, random_state=42)

    train_k_means(config, k_means, rows, rand_uuids, mini_batch_size)

    clustering_labels = apply_k_means(config, k_means, rows, uuids, mini_batch_size)

    evaluation.evaluate_clustering(base_labels, clustering_labels)

    output.result_to_visualize(uuids, base_labels, clustering_labels, clusters)

    model_file = os.path.join(constants.dir_d, constants.dir_mod, 'mb_kmeans_{}_{}.pkl'.format(clusters, rows))
    joblib.dump(k_means, model_file)

    output.out_clustering(dict(zip(uuids, clustering_labels.tolist())), 'euclidean_minibatch', 'kmeans')

    return clustering_labels, k_means


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

    # Divide the docuements in mini batches of fixed size and apply KMeans on them
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


def test_kmeans_clusters(config, base_labels, num_clusters_max, mini_batch_size, rows, uuids, rand_uuids):
    """
    Test several values for the number of clusters.
    Tries 10 different clusterings increasing by 10% of the maximum specified value.

    :param config:global configuration dictionary
    :param base_labels: base labels from AVClass
    :param num_clusters_max: maximum number of clusters to try
    :param mini_batch_size: number of samples per mini batch
    :param rows: number of rows of the data set matrix
    :param uuids: list of documents in order
    :param rand_uuids: list of documents in random order
    :return:
    """

    for mult in np.arange(0.1, 1.1, 0.1):
        cur_num_clusters = int(mult * num_clusters_max)

        if cur_num_clusters >= 2:
            k_means = MiniBatchKMeans(n_clusters=cur_num_clusters, batch_size=mini_batch_size, random_state=42)

            train_k_means(config, k_means, rows, rand_uuids, mini_batch_size)

            clustering_labels = apply_k_means(config, k_means, rows, uuids, mini_batch_size)

            evaluation.evaluate_clustering(base_labels, clustering_labels)

    print('-' * 80)
