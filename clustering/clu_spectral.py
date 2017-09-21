from sklearn.cluster import SpectralClustering
from sklearn.externals import joblib
from utilities import constants
from utilities import output
import utilities.evaluation
import utilities.output
import numpy as np
import os


def cluster(config, num_clusters, uuids, base_labels):
    """
    Cluster the documents using the Jensen-Shannon metric and Spectral Clustering algorithm.

    :return: 
    """

    core_num = config['core_num']

    data = np.loadtxt(os.path.join(constants.dir_d, constants.file_js))

    # Convert distance matrix to affinity matrix
    delta = 1
    data = np.exp(- data ** 2 / (2. * delta ** 2))

    print('Performing clustering')
    spectral = SpectralClustering(affinity='precomputed', n_clusters=num_clusters, n_jobs=core_num, n_init=20)

    clustering_labels = spectral.fit_predict(data)

    utilities.evaluation.evaluate_clustering(base_labels, clustering_labels, data=data, metric='precomputed')

    utilities.output.result_to_visualize(uuids, base_labels, clustering_labels, num_clusters, 'spectral_js')

    output.out_clustering(dict(zip(uuids, clustering_labels.tolist())), 'jensen_shannon', 'spectral')

    model_file = os.path.join(constants.dir_d, constants.dir_mod, 'spectral_{}_{}.pkl'.format('js', len(data)))
    joblib.dump(spectral, model_file)

    return clustering_labels, spectral
