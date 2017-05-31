from workers import wk_read_freqs
from multiprocessing import Pool
from scipy.sparse import vstack
from utilities import utils
import numpy as np


def load_freqs(uuids, core_num, cols, words, dir_malwords, dense=False, ordered=False):
    """
    Load frequency values relative to the specified uuids.

    :param uuids:
    :param core_num:
    :param cols:
    :param words:
    :param dir_malwords:
    :param dense:
    :param ordered:
    :return:
    """

    print('Loading word frequencies of {} documents'.format(len(uuids)))

    file_name_lists = utils.divide_workload(uuids, core_num, ordered)
    formatted_input = utils.format_worker_input(core_num, file_name_lists, (cols, words, dir_malwords, dense))
    pool = Pool(processes=core_num)
    results = pool.map(wk_read_freqs.get_data_matrix, formatted_input)
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

    print(data.shape)

    return data





