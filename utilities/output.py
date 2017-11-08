from utilities import constants
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

    n_samples = len(clustering_dict)

    out_file = 'clustering_{}_{}_{}.json'.format(algo, distance, n_samples)

    json.dump(clustering_dict, open(os.path.join(constants.dir_d, constants.dir_clu, out_file), 'w'), indent=2)


def out_classification(classification_dict, distance, algo):
    """
    Dumps the dictionary resulting from classification into a json file.

    :param classification_dict:
    :param distance:
    :param algo:
    :return:
    """

    n_samples = len(classification_dict)

    out_file = 'classification_{}_{}_{}.json'.format(algo, distance, n_samples)

    json.dump(classification_dict, open(os.path.join(constants.dir_d, constants.dir_cla, out_file), 'w'), indent=2)


def result_to_visualize(uuids, base_labels, clustering_labels, alg=""):
    """
    Generate a json file structured so it can be used for visualization

    :param uuids: list of uuids
    :param base_labels: base truth labels
    :param clustering_labels: clustering results
    :param alg: algorithm used
    :return:
    """

    num_clusters = len(set(clustering_labels)) - (1 if -1 in clustering_labels else 0)

    out_dict = {'name': 'clustering', 'children': []}

    for i in range(num_clusters):
        child_dict = {'name': str(i), 'children': []}

        for j in range(len(clustering_labels)):
            label = int(clustering_labels[j])
            if label == i:
                true_label = int(base_labels[j])
                child_inner = {'name': uuids[j], 'color': true_label}
                child_dict['children'].append(child_inner)

        out_dict['children'].append(child_dict)

    graph_path = os.path.join(constants.dir_d, constants.dir_vis, constants.json_graph.format(alg))
    json.dump(out_dict, open(graph_path, 'w'), indent=2)
