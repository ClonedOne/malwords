from sklearn.cluster import DBSCAN

import utilities.evaluation
from utilities import utils
import numpy as np
import json
import sys
import os

dir_store = ''
core_num = 1
max_iter = 5000


def cluster():
    """
    Cluster the documents using DBScan. 

    :return: 
    """

    global dir_store, core_num
    config = json.load(open('config.json'))
    dir_store = config['dir_store']
    core_num = config['core_num']

    if len(sys.argv) < 2:
        print('Please provide the data matrix file')
        exit()
    matrix_file = sys.argv[1]

    data = np.loadtxt(matrix_file)
    uuids = sorted(os.listdir(dir_store))

    # Retrieve base labels
    print('Acquiring base labels')
    base_labels_dict = utils.get_base_labels()
    base_labels = np.asarray([base_labels_dict[uuid] for uuid in uuids])

    print('Perform DBSCAN')
    dbscan = DBSCAN(n_jobs=core_num, metric='cosine', algorithm='brute')
    computed_labels = dbscan.fit_predict(data)
    num_clusters = len(set(computed_labels)) - (1 if -1 in computed_labels else 0)

    utilities.evaluation.evaluate_clustering(base_labels, computed_labels, data=data)

    utils.result_to_visualize(uuids, base_labels, computed_labels, num_clusters)


if __name__ == '__main__':
    cluster()