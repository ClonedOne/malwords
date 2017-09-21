from sklearn.decomposition import TruncatedSVD
from sklearn.externals import joblib
from helpers import loader_tfidf
from utilities import constants
import numpy as np
import json
import os


def reduce(config, uuids, components):
    """
    Lower dimensionality of data vectors using SVD.

    :param config: configuration dictionary
    :param uuids: list of selected uuids
    :param components: number of desired components
    :return: 
    """

    print('Performing dimensionality reduction using SVD')

    dir_store = config['dir_store']
    core_num = config['core_num']

    svd = TruncatedSVD(n_components=components, n_iter=10)
    words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))

    cols = len(words)
    rows = len(uuids)

    # Force loading of full dataset in RAM (may result in MEMORY ERROR!)
    mini_batch_size = rows

    train_svd(svd, rows, cols, uuids, words, mini_batch_size, core_num, dir_store)

    data = transform_vectors(svd, rows, cols, uuids, words, mini_batch_size, core_num, dir_store)

    print('Explained Variance Ratio')
    print(sum(svd.explained_variance_ratio_))

    matrix_file = os.path.join(constants.dir_d, constants.dir_mat, 'svd_{}_{}.txt'.format(components, rows))
    np.savetxt(open(matrix_file, 'wb'), data)

    model_file = os.path.join(constants.dir_d, constants.dir_mat, 'svd_{}_{}.pkl'.format(components, rows))
    joblib.dump(svd, model_file)

    return data, svd


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
