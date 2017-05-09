from sklearn.cluster import Birch
from utilities import utils
import numpy as np
import json
import sys
import os

dir_store = ''
core_num = 1
branching = 0


def cluster():
    """
    Cluster the documents using Birch algorithm. 

    :return: 
    """

    global dir_store, core_num, branching
    config = json.load(open('config.json'))
    dir_store = config['dir_store']
    core_num = config['core_num']

    if len(sys.argv) < 3:
        print('Please provide the data matrix file and the branching factor')
        exit()
    matrix_file = sys.argv[1]
    branching = int(sys.argv[2])

    data = np.loadtxt(matrix_file)
    uuids = sorted(os.listdir(dir_store))

    # Retrieve base labels
    print('Acquiring base labels')
    base_labels_dict = utils.get_base_labels()
    base_labels = np.asarray([base_labels_dict[uuid] for uuid in uuids])

    print('Perform Birch')
    dbscan = Birch(branching_factor=branching)
    computed_labels = dbscan.fit_predict(data)
    num_clusters = len(set(computed_labels)) - (1 if -1 in computed_labels else 0)

    utils.evaluate_clustering(base_labels, computed_labels, data=data)

    utils.result_to_visualize(uuids, base_labels, computed_labels, num_clusters)


if __name__ == '__main__':
    cluster()
