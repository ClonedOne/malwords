from collections import defaultdict


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
