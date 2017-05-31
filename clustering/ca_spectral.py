from sklearn.cluster import SpectralClustering

import utilities.evaluation
import utilities.output
from utilities import utils
import numpy as np
import json
import sys
import os

dir_store = ''
core_num = 1


def cluster():
    """
    Cluster the documents using out of core Mini Batch KMeans. 

    :return: 
    """

    global dir_store, core_num
    config = json.load(open('config.json'))
    dir_store = config['dir_store']
    core_num = config['core_num']

    if len(sys.argv) < 3:
        print('Please provide the data matrix file and the desired number of clusters')
        exit()

    matrix_file = sys.argv[1]
    num_clusters = int(sys.argv[2])

    data = np.loadtxt(matrix_file)
    uuids = sorted(os.listdir(dir_store))

    # Retrieve base labels
    print('Acquiring base labels')
    base_labels_dict = utils.get_base_labels()
    base_labels = np.asarray([base_labels_dict[uuid] for uuid in uuids])

    print('Perform clustering')
    spectral = SpectralClustering(n_clusters=num_clusters, n_jobs=core_num)
    computed_labels = spectral.fit_predict(data)

    utilities.evaluation.evaluate_clustering(base_labels, computed_labels, data=data)

    utilities.output.result_to_visualize(uuids, base_labels, computed_labels, num_clusters)


if __name__ == '__main__':
    cluster()