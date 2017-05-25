from sklearn.decomposition import TruncatedSVD
from workers import wk_read_tfidf
from multiprocessing import Pool
from utilities import constants
from utilities import utils
from scipy.sparse import *
import numpy as np
import json
import os

dir_store = ''
mini_batch_size = 0
core_num = 1
num_components = 0


def get_svd(config, components):
    """
    Lower dimensionality of data vectors using SVD.

    :return: 
    """

    global dir_store, core_num, mini_batch_size, num_components
    dir_store = config['dir_store']
    core_num = config['core_num']
    num_components = components

    svd = TruncatedSVD(n_components=components)
    words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))
    uuids = sorted(os.listdir(dir_store))

    # Force loading of full dataset in RAM (may result in MEMORY ERROR!)
    mini_batch_size = len(uuids)

    cols = len(words)
    rows = len(uuids)

    decomposed = 0
    transform_vectors(svd, decomposed, rows, cols, uuids, words)

    print('Explained Variance Ratio')
    print(sum(svd.explained_variance_ratio_))


def transform_vectors(svd, decomposed, rows, cols, uuids, words):
    """
    Transform vectors using SVD.    

    :return: 
    """

    # Divide the docuements in mini batches of fixed size and apply Incremental PCA on them
    while decomposed < rows:

        print('Transforming documents from {} to {}'.format(decomposed, (decomposed + mini_batch_size - 1)))
        # starting from the decomposed-th element of the uuids list
        file_name_lists = utils.divide_workload(uuids[decomposed:][:mini_batch_size], core_num, ordered=True)
        formatted_input = utils.format_worker_input(core_num, file_name_lists, (cols, words, dir_store, False))
        pool = Pool(processes=core_num)
        results = pool.map(wk_read_tfidf.get_data_matrix, formatted_input)
        pool.close()
        pool.join()

        # sort results
        acc = []
        for i in range(core_num):
            for res in results:
                if res[0] == i:
                    acc.append(res[1])
        data = vstack(acc)

        print(data.shape)

        del results
        del acc
        decomposed += mini_batch_size

        new_data = svd.fit_transform(data)

        matrix_file = os.path.join(constants.dir_d, constants.dir_dm, "svd_{}.txt".format(num_components))
        np.savetxt(open(matrix_file, "ab"), new_data)
