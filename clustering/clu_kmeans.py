from utilities import output, constants
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from utilities import interaction
from utilities import evaluation
from helpers import loader_tfidf
from operator import itemgetter
import numpy as np
import json
import os


def cluster(config, data_matrix, clusters, uuids, base_labels, sparse=False):
    """
    Cluster the documents using KMeans algorithm. 

    :return: 
    """

    max_iter = 3000
    core_num = config['core_num']
    dir_store = config['dir_store']
    words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))

    num_clusters_max = clusters

    if not sparse:
        data = np.loadtxt(data_matrix)
    else:
        data = loader_tfidf.load_tfidf(uuids, core_num, len(words), words, dir_store, dense=False, ordered=True)

    test_kmeans_clusters(data, base_labels, num_clusters_max, core_num, max_iter)

    # Forces the user to choose the desired number of clusters
    num_clusters = interaction.ask_clusters(num_clusters_max)

    k_means = KMeans(n_clusters=num_clusters, n_jobs=core_num, max_iter=max_iter, verbose=1)
    clustering_labels = k_means.fit_predict(data)

    evaluation.evaluate_clustering(base_labels, clustering_labels, data=data)

    output.out_clustering(dict(zip(uuids, clustering_labels.tolist())), 'euclidean', 'kmeans')

    output.result_to_visualize(uuids, base_labels, clustering_labels, num_clusters)

    model_file = os.path.join(constants.dir_d, constants.dir_mod, 'kmeans_{}_{}.pkl'.format(num_clusters, len(data)))
    joblib.dump(k_means, model_file)

    return clustering_labels, k_means


def test_kmeans_clusters(data, base_labels, num_clusters_max, core_num, max_iter):
    """
    Test several values for the number of clusters. 
    Tries 10 different clusterings increasing by 10% of the maximum specified value.
    
    :param data: data matrix to cluster
    :param base_labels: base labels from AVClass
    :param num_clusters_max: maximum number of clusters to try
    :param core_num: number of available cpu cores
    :param max_iter: maximum number of iterations
    :return:
    """

    silhouettes = {}

    for mult in np.arange(0.1, 1.1, 0.1):
        cur_num_clusters = int(mult * num_clusters_max)

        if cur_num_clusters >= 2:
            k_means = KMeans(n_clusters=cur_num_clusters, n_jobs=core_num, max_iter=max_iter)
            computed_labels = k_means.fit_predict(data)

            ars, ami, fm, h, c, p, r, fs, sh = evaluation.evaluate_clustering(base_labels, computed_labels, data=data)
            silhouettes[cur_num_clusters] = sh

    print('-' * 80)
    for sh in sorted(silhouettes.items(), key=itemgetter(1), reverse=True):
        print('Clusters {} Silhouette {}'.format(sh[0], sh[1]))
