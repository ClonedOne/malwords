from sklearn.metrics.pairwise import pairwise_distances
from workers import wk_read_freqs, wk_read_tfidf
from utilities import utils, constants
from distances import jensen_shannon
from multiprocessing import Pool
from scipy.sparse import *
import numpy as np
import json
import os


def get_js(config):
    """
    Produces a distance matrix applying the Jensen-Shannon Distance to all the feature vectors.
    
    :param config: 
    :return: 
    """

    dir_store = config['dir_store']
    core_num = config['core_num']

    uuids = sorted(os.listdir(dir_store))
    words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))
    cols = len(words)

    print('Creating Jensen-Shannon distance matrix')

    # Force loading of full data-set in RAM (may be a problem with low memory!)
    mini_batch_size = len(uuids)
    decomposed = 0

    file_name_lists = utils.divide_workload(uuids[decomposed:][:mini_batch_size], core_num, ordered=True)
    formatted_input = utils.format_worker_input(core_num, file_name_lists, (cols, words, dir_store, False))
    pool = Pool(processes=core_num)
    results = pool.map(wk_read_tfidf.get_data_matrix, formatted_input)
    pool.close()
    pool.join()

    # # Sort results by their worker number
    # acc = []
    # # Each worker will return a list of size (mini_batch_size / core_num)
    # for i in range(core_num):
    #     for res in results:
    #         if res[0] == i:
    #             acc.append(res[1])
    # data = np.concatenate(acc)
    # print(data.nbytes)

    # sort results
    acc = []
    for i in range(core_num):
        for res in results:
            if res[0] == i:
                acc.append(res[1])
    data = vstack(acc)
    data = data.tocsr()
    print(data.data.nbytes)

    print(data.shape)
    print(data.sum())

    # distances = pairwise_distances(data, metric=jensen_shannon.compute_js_dist, n_jobs=core_num)
    # matrix_file = os.path.join(constants.dir_d, constants.file_js)
    # np.savetxt(matrix_file, distances)
