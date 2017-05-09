from sklearn.cluster import DBSCAN
from utilities import utils
from sklearn import metrics
import numpy as np
import json
import os

dir_store = ''
core_num = 1
max_iter = 1000


def cluster():
    """
    Cluster the documents using out of core Mini Batch KMeans. 

    :return: 
    """

    global dir_store, core_num
    config = json.load(open('config.json'))
    dir_store = config['dir_store']
    core_num = config['core_num']
    uuids = sorted(os.listdir(dir_store))

    matrix_file = open('data/matrix.txt', 'r')
    data = np.loadtxt(matrix_file)

    # Retrieve base labels
    print('Acquire base labels')
    base_labels = utils.get_base_labels_old(uuids)
    base_labels = np.asarray(base_labels)

    print('Perform DBSCAN')
    dbscan = DBSCAN(n_jobs=core_num, metric='jaccard')
    computed_labels = dbscan.fit_predict(data)
    num_clusters = len(set(computed_labels)) - (1 if -1 in computed_labels else 0)

    utils.evaluate_clustering(base_labels, computed_labels, data=data)

    utils.result_to_visualize(uuids, base_labels, computed_labels, num_clusters)


if __name__ == '__main__':
    cluster()
