from sklearn.decomposition import IncrementalPCA
from multiprocessing import Pool
from utilities import utils
import numpy as np
import random
import json
import os


dir_store = ''
mini_batch_size = 0
core_num = 1


def get_pca():
    """
    Apply Incremental Principal Components Analysis to the tf-idf vectors.
    
    :return: 
    """

    global dir_store, core_num, mini_batch_size
    config = json.load(open('config.json'))
    dir_store = config['dir_store']
    core_num = config['core_num']
    mini_batch_size = config['batch_size']

    i_pca = IncrementalPCA(n_components=2, batch_size=mini_batch_size)
    words = json.load(open('data/words.json', 'r'))
    uuids = sorted(os.listdir(dir_store))
    rand_uuids = random.sample(uuids, len(uuids))

    cols = len(words)
    rows = len(uuids)

    decomposed = 0
    train_pca(i_pca, decomposed, rows, cols, rand_uuids, words)

    decomposed = 0
    transform_vectors(i_pca, decomposed, rows, cols, uuids, words)


def train_pca(i_pca, decomposed, rows, cols, rand_uuids, words):
    """
    Train the PCA algorithm incrementally using mini batches of data.    
    
    :return: 
    """

    # Divide the docuements in mini batches of fixed size and apply Incremental PCA on them
    while decomposed < rows:

        print('Processing documents from {} to {}'.format(decomposed, (decomposed + mini_batch_size - 1)))
        # starting from the decomposed-th element of the uuids list
        file_name_lists = utils.divide_workload(rand_uuids[decomposed:][:mini_batch_size], core_num)
        formatted_input = utils.format_worker_input(core_num, file_name_lists, (cols, words))
        pool = Pool(processes=core_num)
        results = pool.map(get_data_matrix, formatted_input)
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
    Train the PCA algorithm incrementally using mini batches of data.    

    :return: 
    """

    # Divide the docuements in mini batches of fixed size and apply Incremental PCA on them
    while decomposed < rows:

        print('Transforming documents from {} to {}'.format(decomposed, (decomposed + mini_batch_size - 1)))
        # starting from the decomposed-th element of the uuids list
        file_name_lists = utils.divide_workload(uuids[decomposed:][:mini_batch_size], core_num, ordered=True)
        formatted_input = utils.format_worker_input(core_num, file_name_lists, (cols, words))
        pool = Pool(processes=core_num)
        results = pool.map(get_data_matrix, formatted_input)
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
        np.savetxt(open("data/matrix2d.txt", "ab"), new_data)


def extract_tf_idf(tf_idf_file, words):
    """
    Construct an iterator over non zero tf-idf values for each word in the words list

    :param tf_idf_file: path to the file containing the bag of words with tf-idf 
    :param words: ordered list of words
    :return: iterator over non zero tf-idf values
    """

    tf_idf_json = json.load(open(tf_idf_file, 'r'))
    for word in tf_idf_json:
        word_index = words[word]
        tf_idf = tf_idf_json[word]
        yield (word_index, tf_idf)


def get_data_matrix(data_pack):
    """
    Computes the dense matrix used as input for the Incremental PCA algorithm.
    The data pack contains:

     * number of rows
     * number of columns
     * list of uuids
     * dictionary of words and their positional index

    :param data_pack: input data for the worker process
    :return: sparse tf-idf matrix
    """

    # Unpacking data from main process
    process_id = data_pack[0]
    uuids = data_pack[1]
    rows = len(uuids)
    cols = data_pack[2]
    words = data_pack[3]

    # print(len(uuids), rows, cols, len(words))

    data = np.zeros((rows, cols))

    # Generates dense matrix from tf-idf vector files
    row = 0
    for uuid in uuids:
        for (col, tf_idf) in extract_tf_idf(os.path.join(dir_store, uuid), words):
            data[row, col] = tf_idf

        row += 1

    return process_id, data


if __name__ == '__main__':
    get_pca()
