from collections import defaultdict
import random
import json


def divide_workload(item_list, core_num, ordered=False):
    """
    Given a list of items and the number of CPU cores available, computes equal sized lists of items for each core. 

    :param item_list: list of items to split
    :param core_num: number of available CPU cores (workers)
    :param ordered: flag if set result should be ordered as the incoming list
    :return: defaultdict containing lists of elements divided equally
    """

    j = 0
    c = 0
    item_sublists = defaultdict(list)

    if not ordered:
        for item in item_list:
            item_sublists[j].append(item)
            j = (j + 1) % core_num
    else:
        per_core = int(len(item_list) / core_num)
        extra = len(item_list) % core_num

        for c in range(core_num):
            item_sublists[c] = (item_list[j:(j + per_core)])
            j += per_core

        if extra:
            item_sublists[c] += (item_list[j:])

    if len(item_list) < core_num:
        while j < core_num:
            item_sublists[j] = []
            j += 1

    if len(item_sublists) != core_num:
        print('Error: size of split workload different from number of cores')
        quit()

    return item_sublists


def format_worker_input(core_num, item_sublists, fixed_params):
    """
    Generate a list of tuples containing the parameters to pass to worker sub processes.

    :param core_num: number of available cores
    :param item_sublists: dictionary containing the sublist of files for each worker
    :param fixed_params: list of parameters to be added to workers input
    :return: formatted list of worker input parameters
    """

    formatted_input = []
    for i in range(core_num):
        formatted_input.append((i, item_sublists[i]) + tuple(fixed_params))
    return formatted_input


def get_base_labels(uuids):
    """
    Returns the ordered list of base labels from AVClass output

    :return: ordered list of labels
    """

    base_labels = []
    inverted_label = json.load(open('data/inverted_labels.json'))
    uuid_label = json.load(open('data/labels.json'))
    sorted_families = sorted(list(inverted_label.keys()))
    families_index = {sorted_families[index]: index for index in range(len(sorted_families))}

    for uuid in uuids:
        base_labels.append(families_index[uuid_label[uuid]])

    return base_labels


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

    graph_path = 'visualize/graph1.json'
    json.dump(out_dict, open(graph_path, 'w'), indent=2)
