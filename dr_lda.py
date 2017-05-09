from sklearn.decomposition import LatentDirichletAllocation
from multiprocessing import Pool
from workers import wk_dense
from utilities import utils
import numpy as np
import random
import json
import sys
import os

dir_store = ''
mini_batch_size = 0
core_num = 1
components = 0


def get_lda():
    """
    Apply Incremental Principal Components Analysis to the tf-idf vectors.

    :return: 
    """

    global dir_store, core_num, mini_batch_size, components
    config = json.load(open('config.json'))
    dir_store = config['dir_store']
    core_num = config['core_num']
    mini_batch_size = config['batch_size']

    if len(sys.argv) < 2:
        print('Specify number of components')
        exit()
    components = int(sys.argv[1])

    words = json.load(open('data/words.json', 'r'))
    uuids = sorted(os.listdir(dir_store))
    rand_uuids = random.sample(uuids, len(uuids))

    cols = len(words)
    rows = len(uuids)

    lda = LatentDirichletAllocation(batch_size=mini_batch_size, n_jobs=core_num, n_topics=components, max_iter=50,
                                    total_samples=len(uuids), random_state=42)

    decomposed = 0
    train_lda(lda, decomposed, rows, cols, rand_uuids, words)

    decomposed = 0
    transform_vectors(lda, decomposed, rows, cols, uuids, words)


def train_lda(lda, decomposed, rows, cols, rand_uuids, words):
    """
    Train the LDA algorithm incrementally using mini batches of data.    

    :return: 
    """

    while decomposed < rows:

        print('Processing documents from {} to {}'.format(decomposed, (decomposed + mini_batch_size - 1)))
        # starting from the decomposed-th element of the uuids list
        file_name_lists = utils.divide_workload(rand_uuids[decomposed:][:mini_batch_size], core_num)
        formatted_input = utils.format_worker_input(core_num, file_name_lists, (cols, words, dir_store))
        pool = Pool(processes=core_num)
        results = pool.map(wk_dense.get_data_matrix, formatted_input)
        pool.close()
        pool.join()

        # sort results
        acc = []
        # Each worker will return a list of size (mini_batch_size / core_num)
        for i in range(core_num):
            for res in results:
                if res[0] == i:
                    acc.append(res[1])
        data = np.concatenate(acc)

        print(data.shape)

        del results
        del acc
        decomposed += mini_batch_size

        lda.partial_fit(data)


def transform_vectors(i_pca, decomposed, rows, cols, uuids, words):
    """
    Transorm the data vectors in mini batches.   

    :return: 
    """

    while decomposed < rows:

        print('Transforming documents from {} to {}'.format(decomposed, (decomposed + mini_batch_size - 1)))
        # starting from the decomposed-th element of the uuids list
        file_name_lists = utils.divide_workload(uuids[decomposed:][:mini_batch_size], core_num, ordered=True)
        formatted_input = utils.format_worker_input(core_num, file_name_lists, (cols, words, dir_store))
        pool = Pool(processes=core_num)
        results = pool.map(wk_dense.get_data_matrix, formatted_input)
        pool.close()
        pool.join()

        # sort results
        acc = []
        # Each worker will return a list of size (mini_batch_size / core_num)
        for i in range(core_num):
            for res in results:
                if res[0] == i:
                    acc.append(res[1])
        data = np.concatenate(acc)

        print(data.shape)

        del results
        del acc
        decomposed += mini_batch_size

        new_data = i_pca.transform(data)

        matrix_file = "data/matrix_lda_{}.txt".format(components)
        np.savetxt(open(matrix_file, "ab"), new_data)


if __name__ == '__main__':
    get_lda()
