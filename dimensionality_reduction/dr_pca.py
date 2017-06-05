from sklearn.decomposition import IncrementalPCA
from workers import wk_read_tfidf
from multiprocessing import Pool
from utilities import constants
from utilities import utils
import numpy as np
import random
import json
import os


dir_store = ''
mini_batch_size = 0
core_num = 1
num_components = 0


def get_pca(config, components, uuids):
    """
    Apply Incremental Principal Components Analysis to the tf-idf vectors.
    
    :return: 
    """

    global dir_store, core_num, mini_batch_size, num_components
    dir_store = config['dir_store']
    core_num = config['core_num']
    mini_batch_size = config['batch_size']
    num_components = components

    i_pca = IncrementalPCA(n_components=num_components, batch_size=mini_batch_size)
    words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))
    rand_uuids = random.sample(uuids, len(uuids))

    cols = len(words)
    rows = len(uuids)

    decomposed = 0
    train_pca(i_pca, decomposed, rows, cols, rand_uuids, words)

    print('Explained Variance Ratio')
    print(sum(i_pca.explained_variance_ratio_))

    decomposed = 0
    transform_vectors(i_pca, decomposed, rows, cols, uuids, words)


def train_pca(i_pca, decomposed, rows, cols, rand_uuids, words):
    """
    Train the PCA algorithm incrementally using mini batches of data.    
    
    :return: 
    """

    while decomposed < rows:

        print('Processing documents from {} to {}'.format(decomposed, (decomposed + mini_batch_size - 1)))
        # starting from the decomposed-th element of the uuids list
        file_name_lists = utils.divide_workload(rand_uuids[decomposed:][:mini_batch_size], core_num)
        formatted_input = utils.format_worker_input(core_num, file_name_lists, (cols, words, dir_store, True))
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
        data = np.concatenate(acc)

        print(data.shape)

        del results
        del acc
        decomposed += mini_batch_size

        i_pca.partial_fit(data)


def transform_vectors(i_pca, decomposed, rows, cols, uuids, words):
    """
    Transorm the data vectors in mini batches.  

    :return: 
    """

    while decomposed < rows:

        print('Transforming documents from {} to {}'.format(decomposed, (decomposed + mini_batch_size - 1)))
        # starting from the decomposed-th element of the uuids list
        file_name_lists = utils.divide_workload(uuids[decomposed:][:mini_batch_size], core_num, ordered=True)
        formatted_input = utils.format_worker_input(core_num, file_name_lists, (cols, words, dir_store, True))
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
        data = np.concatenate(acc)

        print(data.shape)

        del results
        del acc
        decomposed += mini_batch_size

        new_data = i_pca.transform(data)

        matrix_file = os.path.join(constants.dir_d, constants.dir_dm, "pca_{}.txt".format(num_components))
        np.savetxt(open(matrix_file, "ab"), new_data)
