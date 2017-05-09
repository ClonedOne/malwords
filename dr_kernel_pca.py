from sklearn.decomposition import KernelPCA
from multiprocessing import Pool
from workers import wk_sparse
from utilities import utils
from scipy.sparse import *
import numpy as np
import json
import sys
import os

dir_store = ''
mini_batch_size = 0
core_num = 1
components = 0


def get_kern_pca():
    """
    Lower dimensionality of data vectors using tSNE.

    :return: 
    """

    global dir_store, core_num, mini_batch_size, components
    config = json.load(open('config.json'))
    dir_store = config['dir_store']
    core_num = config['core_num']

    if len(sys.argv) < 2:
        print('Specify number of components')
        exit()
    components = int(sys.argv[1])

    kernel_pca = KernelPCA(n_components=components, random_state=42, n_jobs=core_num, kernel='cosine', remove_zero_eig=True, copy_X=False)
    words = json.load(open('data/words.json', 'r'))
    uuids = sorted(os.listdir(dir_store))

    # Force loading of full dataset in RAM (may be a problem with low memory!)
    mini_batch_size = len(uuids)

    cols = len(words)
    rows = len(uuids)

    decomposed = 0
    transform_vectors(kernel_pca, decomposed, rows, cols, uuids, words)


def transform_vectors(kernel_pca, decomposed, rows, cols, uuids, words):
    """
    Transform vectors using Kernel PCA    

    :return: 
    """

    # Divide the docuements in mini batches of fixed size and apply Incremental PCA on them
    while decomposed < rows:

        print('Transforming documents from {} to {}'.format(decomposed, (decomposed + mini_batch_size - 1)))
        # starting from the decomposed-th element of the uuids list
        file_name_lists = utils.divide_workload(uuids[decomposed:][:mini_batch_size], core_num, ordered=True)
        formatted_input = utils.format_worker_input(core_num, file_name_lists, (cols, words, dir_store))
        pool = Pool(processes=core_num)
        results = pool.map(wk_sparse.get_data_matrix, formatted_input)
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

        new_data = kernel_pca.fit_transform(data)

        matrix_file = "data/matrix_kern_pca_{}.txt".format(components)
        np.savetxt(open(matrix_file, "ab"), new_data)


if __name__ == '__main__':
    get_kern_pca()
