from workers import wk_read_tfidf
from multiprocessing import Pool
from scipy.sparse import vstack
from utilities import constants
from utilities import utils
import numpy as np
import json
import os


def load_tfidf(config, uuids, dense=False, ordered=False):
    """
    Load tf-idf values relative to the specified uuids.

    :param config: Global configuration dictionary
    :param uuids: List of files to read
    :param dense: Flag, if set return a dense matrix, else return a sparse matrix
    :param ordered: Falg, if set return a matrix whose rows are ordered as the uuids list
    :return:
    """

    core_num = config['core_num']
    dir_store = config['dir_store']
    words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))
    cols = len(words)

    print('Loading Tf-Idf of {} documents'.format(len(uuids)))

    file_name_lists = utils.divide_workload(uuids, core_num, ordered)
    formatted_input = utils.format_worker_input(core_num, file_name_lists, (cols, words, dir_store, dense))
    pool = Pool(processes=core_num)
    results = pool.map(wk_read_tfidf.get_data_matrix, formatted_input)
    pool.close()
    pool.join()

    # sort results
    acc = []
    # Each worker will return a list of size (mini_batch_size / core_num)
    for i in range(core_num):
        for res in results:
            if res[0] == i:
                acc.append(res[1])

    if dense:
        data = np.concatenate(acc)
    else:
        data = vstack(acc)
        data = data.tocsc()

    print(data.shape)

    return data
