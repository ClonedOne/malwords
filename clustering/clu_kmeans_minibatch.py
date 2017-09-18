from sklearn.cluster import MiniBatchKMeans
from helpers import loader_tfidf
from utilities import evaluation
from utilities import output
from scipy.sparse import *
import numpy as np
import random
import json


def cluster(config, clusters, uuids, base_labels):
    """
    Cluster the documents using out of core Mini Batch KMeans. 
    
    :return: 
    """

    dir_store = config['dir_store']
    core_num = config['core_num']
    mini_batch_size = config['batch_size']
    num_clusters = clusters

    k_means = MiniBatchKMeans(n_clusters=num_clusters, batch_size=mini_batch_size)
    words = json.load(open('data/words.json', 'r'))
    rand_uuids = random.sample(uuids, len(uuids))

    cols = len(words)
    rows = len(uuids)

    print('Matrix dimensions: ', rows, cols)

    print('\nTraining KMeans')
    train_k_means(k_means, rows, cols, rand_uuids, words, mini_batch_size, core_num, dir_store)

    print('\nPredicting values')
    clustering_labels = apply_k_means(k_means, rows, cols, uuids, words, mini_batch_size, core_num, dir_store)

    evaluation.evaluate_clustering(base_labels, clustering_labels)

    output.result_to_visualize(uuids, base_labels, clustering_labels, num_clusters)

    return clustering_labels, k_means


def train_k_means(k_means, rows, cols, rand_uuids, words, mini_batch_size, core_num, dir_store):
    """
    Train the MiniBatchKMeans clustering algorithm with fixed size data batches taken random from the input data-set. 
    
    :param dir_store:
    :param core_num:
    :param mini_batch_size:
    :param k_means: MiniBatchKMeans object
    :param rows: number of rows of the data set matrix
    :param cols: number of columns of the data set matrix
    :param rand_uuids: list of documents in randomized order
    :param words: dictionary mapping word features to their index
    :return: 
    """

    clustered = 0

    # Divide the docuements in mini batches of fixed size and train the KMeans
    while clustered < rows:
        print('Processing documents from {} to {}'.format(clustered, (clustered + mini_batch_size - 1)))

        data = loader_tfidf.load_tfidf(rand_uuids[clustered:][:mini_batch_size], core_num, cols, words, dir_store,
                                       dense=True, ordered=False)

        clustered += mini_batch_size

        k_means.partial_fit(data)


def apply_k_means(k_means, rows, cols, uuids, words, mini_batch_size, core_num, dir_store):
    """
    Apply the MiniBatchKMeans clustering algorithm with fixed size data batches taken in order from the input data set. 

    :param mini_batch_size:
    :param core_num:
    :param dir_store:
    :param k_means: MiniBatchKMeans object
    :param rows: number of rows of the data set matrix
    :param cols: number of columns of the data set matrix
    :param uuids: list of documents in order
    :param words: dictionary mapping word features to their index
    :return: labels computed by KMeans over the data set
    """

    clustered = 0
    computed_labels = np.array([])

    # Divide the docuements in mini batches of fixed size and apply KMeans on them
    while clustered < rows:
        print('Predicting documents from {} to {}'.format(clustered, (clustered + mini_batch_size - 1)))

        data = loader_tfidf.load_tfidf(uuids[clustered:][:mini_batch_size], core_num, cols, words, dir_store,
                                       dense=True, ordered=True)

        clustered += mini_batch_size

        batch_computed_labels = k_means.predict(data)

        computed_labels = np.append(computed_labels, batch_computed_labels)

    return computed_labels
