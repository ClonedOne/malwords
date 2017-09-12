from collections import defaultdict
from utilities import constants
import numpy as np
import json
import os


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


def get_base_labels_uuids(uuids):
    """
    Returns a list of numerical indices mapping each uuid to a malware family.

    :param uuids: list of uuids whose malware families are required
    :return: list of malware families
    """

    base_labels_dict = get_base_labels()
    base_labels = np.asarray([base_labels_dict[uuid] for uuid in uuids])

    return base_labels


def get_base_labels():
    """
    Returns a dictionary mapping uuids to its malware family expressed as numerical index. 
    
    :return: ordered list of labels
    """

    base_labels = {}
    inverted_label = json.load(open(os.path.join(constants.dir_d, constants.json_inverted_labels)))
    uuid_label = json.load(open(os.path.join(constants.dir_d, constants.json_labels)))
    sorted_families = sorted(list(inverted_label.keys()))
    families_index = {sorted_families[index]: index for index in range(len(sorted_families))}

    for uuid in uuid_label:
        base_labels[uuid] = families_index[uuid_label[uuid]]

    return base_labels


def get_base_words(dir_base):
    """
    Returns a set containing the words recorded with a clean Malrec session
    
    :return: set of words
    """

    base_file = os.path.join(dir_base, 'base.txt')
    base_words = set()

    if os.path.isfile(base_file):
        with open(base_file, 'rb') as base_in:
            for line in base_in:
                base_words.add(line.strip().split()[0].decode('utf-8'))

    return base_words


def get_index_labels():
    """
    Returns a dictionary mapping numerical index to malware families.

    :return: dictionary mapping integers to malware families
    """

    inverted_label = json.load(open(os.path.join(constants.dir_d, constants.json_inverted_labels)))
    sorted_families = sorted(list(inverted_label.keys()))
    index_label = {index: sorted_families[index] for index in range(len(sorted_families))}

    return index_label


def get_all_uuids(file_dir):
    """
    Returns a sorted list of all the uuids in the malwords path.

    :param file_dir: path where files are located
    :return: sorted list of uuids
    """

    if not os.path.isdir(file_dir):
        print('Could not find malwords directory')
        exit()

    return [uuid.strip().split('_')[0] for uuid in sorted(os.listdir(file_dir))]
