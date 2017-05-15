import json
import os


def out_clustering(clustering_dict, distance, algo):
    """
    Dumps the dictionary resulting from clustering into a json file.
    
    :param clustering_dict: result of clustering algorithm
    :param distance: name of distance metric used 
    :param algo: name of clustering algorithm 
    :return: 
    """

    dest = 'clustering'

    if not os.path.exists(dest):
        os.makedirs(dest)

    n_samples = len(clustering_dict)

    out_file = 'clustering_{}_{}_{}.json'.format(algo, distance, n_samples)

    json.dump(clustering_dict, open(os.path.join(dest, out_file), 'w'), indent=2)
