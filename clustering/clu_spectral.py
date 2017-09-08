from sklearn.cluster import SpectralClustering
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

    computed_labels = spectral.fit_predict(data)

    utilities.evaluation.evaluate_clustering(base_labels, computed_labels, data=data, metric='precomputed')

    utilities.output.result_to_visualize(uuids, base_labels, computed_labels, num_clusters, 'spectral_js')

    output.out_clustering(dict(zip(uuids, computed_labels.tolist())), 'jensen_shannon', 'spectral')
