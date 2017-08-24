from sklearn.decomposition import TruncatedSVD
from helpers import loader_tfidf
from utilities import constants
import numpy as np
import json
import os


def reduce(config, train, test, components, objective):
    """
    Lower dimensionality of data vectors using SVD.

    :return: 
    """

    print('Performing dimensionality reduction using SVD')

    dir_store = config['dir_store']
    core_num = config['core_num']

    svd = TruncatedSVD(n_components=components, n_iter=10)
    words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))

    # Force loading of full dataset in RAM (may result in MEMORY ERROR!)
    mini_batch_size = len(train)

    cols = len(words)
    rows = len(train)

    train_svd(svd, rows, cols, train, words, mini_batch_size, core_num, dir_store)

    data = transform_vectors(svd, rows, cols, train, words, mini_batch_size, core_num, dir_store)

    print('Explained Variance Ratio')
    print(sum(svd.explained_variance_ratio_))

    matrix_file = os.path.join(constants.dir_d, constants.dir_dm, "svd_{}_{}.txt".format(components, objective))
    np.savetxt(open(matrix_file, "wb"), data)

    if test is not None:
        rows = len(test)
        mini_batch_size = len(test)

        data = transform_vectors(svd, rows, cols, test, words, mini_batch_size, core_num, dir_store)

        matrix_file = os.path.join(constants.dir_d, constants.dir_dm, "svd_{}_{}.txt".format(components, 'test'))
        np.savetxt(open(matrix_file, "wb"), data)


def train_svd(svd, rows, cols, rand_uuids, words, mini_batch_size, core_num, dir_store):
    """
    Train the SVD algorithm.

    :return:
    """

    decomposed = 0

    while decomposed < rows:
        print('Processing documents from {} to {}'.format(decomposed, (decomposed + mini_batch_size - 1)))
        data = loader_tfidf.load_tfidf(rand_uuids[decomposed:][:mini_batch_size], core_num, cols, words, dir_store,
                                       dense=False, ordered=False)

        decomposed += mini_batch_size

        svd.fit(data)


def transform_vectors(svd, rows, cols, uuids, words, mini_batch_size, core_num, dir_store):
    """
    Transform vectors using SVD.    

    :return: 
    """

    decomposed = 0
    new_data = []

    while decomposed < rows:
        print('Transforming documents from {} to {}'.format(decomposed, (decomposed + mini_batch_size - 1)))

        data = loader_tfidf.load_tfidf(uuids[decomposed:][:mini_batch_size], core_num, cols, words, dir_store,
                                       dense=False, ordered=True)

        decomposed += mini_batch_size

        new_data.append(svd.transform(data))

    return np.concatenate(new_data)
