from sklearn.cluster import DBSCAN
from utilities import evaluation
from utilities import constants
from utilities import output
from utilities import utils
import numpy as np
import os


def cluster(config):
    """
    Cluster the documents using DBScan. 

    :return: 
    """

    dir_store = config['dir_store']
    core_num = config['core_num']

    uuids = sorted(os.listdir(dir_store))
    data = np.loadtxt(os.path.join(constants.dir_d, constants.file_js))

    # Retrieve base labels
    print('Acquiring base labels')
    base_labels_dict = utils.get_base_labels()
    base_labels = np.asarray([base_labels_dict[uuid] for uuid in uuids])

    print('Perform clustering')
    dbscan = DBSCAN(eps=0.55, metric='precomputed', n_jobs=core_num, min_samples=25)
    computed_labels = dbscan.fit_predict(data)
    num_clusters = len(set(computed_labels)) - (1 if -1 in computed_labels else 0)

    if num_clusters == 1:
        data = None

    evaluation.evaluate_clustering(base_labels, computed_labels, data=data, metric='precomputed')

    output.result_to_visualize(uuids, base_labels, computed_labels, num_clusters)

    output.out_clustering(dict(zip(uuids, computed_labels.tolist())), 'jensen_shannon', 'dbscan')
