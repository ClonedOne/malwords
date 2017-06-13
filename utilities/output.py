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

    json.dump(clustering_dict, open(os.path.join(constants.dir_d, constants.dir_dc, out_file), 'w'), indent=2)


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

    json.dump(classification_dict, open(os.path.join(constants.dir_d, constants.dir_ds, out_file), 'w'), indent=2)


def result_to_visualize(uuids, base_labels, computed_labels, num_clusters):
    """
    Generate a json file structured so it can be used for visualization

    :param uuids: list of uuids
    :param base_labels: base truth labels
    :param computed_labels: clustering results
    :param num_clusters: number of clusters created
    :return:
    """

    out_dict = {'name': 'clustering', 'children': []}

    for i in range(num_clusters):
        child_dict = {'name': str(i), 'children': []}

        for j in range(len(computed_labels)):
            label = int(computed_labels[j])
            if label == i:
                true_label = int(base_labels[j])
                child_inner = {'name': uuids[j], 'color': true_label}
                child_dict['children'].append(child_inner)

        out_dict['children'].append(child_dict)

    graph_path = os.path.join(constants.dir_d, constants.dir_dv, constants.json_graph)
    json.dump(out_dict, open(graph_path, 'w'), indent=2)
