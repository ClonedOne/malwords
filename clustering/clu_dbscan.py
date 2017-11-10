from sklearn.externals import joblib
from sklearn.cluster import DBSCAN
from utilities import evaluation
from utilities import constants
from utilities import output
import numpy as np
import os


def cluster(config, uuids, base_labels):
    """
    Cluster the documents using DBScan. 

    :return: 
    """

    core_num = config['core_num']

    data = np.loadtxt(os.path.join(constants.dir_d, constants.file_js))

    dbscan = DBSCAN(eps=0.55, metric='precomputed', n_jobs=core_num, min_samples=25)
    clustering_labels = dbscan.fit_predict(data)
    num_clusters = len(set(clustering_labels)) - (1 if -1 in clustering_labels else 0)

    if num_clusters == 1:
        data = None

    evaluation.evaluate_clustering(base_labels, clustering_labels, data=data, metric='precomputed')

    output.result_to_visualize(uuids, base_labels, clustering_labels, num_clusters, 'dbscan_js')

    output.out_clustering(dict(zip(uuids, clustering_labels.tolist())), 'jensen_shannon', 'dbscan')

    model_file = os.path.join(constants.dir_d, constants.dir_mod, 'dbscan_{}_{}.pkl'.format('js', len(data)))
    joblib.dump(dbscan, model_file)

    return clustering_labels, dbscan
