from sklearn.decomposition import IncrementalPCA
from helpers import loader_tfidf
from utilities import constants
import numpy as np
import random
import json
import os


def reduce(config, train, test, components, objective):
    """
    Apply Incremental Principal Components Analysis to the tf-idf vectors.
    
    :return: 
    """

    print('Performing dimensionality reduction using PCA')

    dir_store = config['dir_store']
    core_num = config['core_num']
    mini_batch_size = config['batch_size']

    words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))
    rand_train = random.sample(train, len(train))

    cols = len(words)
    rows = len(train)

    i_pca = IncrementalPCA(n_components=components, batch_size=mini_batch_size)

    train_pca(i_pca, rows, cols, rand_train, words, mini_batch_size, core_num, dir_store)

    print('Explained Variance Ratio')
    print(sum(i_pca.explained_variance_ratio_))

    data = transform_vectors(i_pca, rows, cols, train, words, mini_batch_size, core_num, dir_store)

    matrix_file = os.path.join(constants.dir_d, constants.dir_dm, "pca_{}_{}.txt".format(components, objective))
    np.savetxt(open(matrix_file, "wb"), data)

    if test is not None:
        rows = len(test)
        data = transform_vectors(i_pca, rows, cols, test, words, mini_batch_size, core_num, dir_store)

        matrix_file = os.path.join(constants.dir_d, constants.dir_dm, "pca_{}_{}.txt".format(components, 'test'))
        np.savetxt(open(matrix_file, "wb"), data)


def train_pca(i_pca, rows, cols, rand_uuids, words, mini_batch_size, core_num, dir_store):
    """
    Train the PCA algorithm incrementally using mini batches of data.    
    
    :return: 
    """

    decomposed = 0

    while decomposed < rows:
        print('Processing documents from {} to {}'.format(decomposed, (decomposed + mini_batch_size - 1)))
        data = loader_tfidf.load_tfidf(rand_uuids[decomposed:][:mini_batch_size], core_num, cols, words, dir_store,
                                       dense=True, ordered=False)

        decomposed += mini_batch_size

        i_pca.partial_fit(data)


def transform_vectors(i_pca, rows, cols, uuids, words, mini_batch_size, core_num, dir_store):
    """
    Transorm the data vectors.

    :return: 
    """

    decomposed = 0
    new_data = []

    while decomposed < rows:
        print('Transforming documents from {} to {}'.format(decomposed, (decomposed + mini_batch_size - 1)))
        data = loader_tfidf.load_tfidf(uuids[decomposed:][:mini_batch_size], core_num, cols, words, dir_store,
                                       dense=True, ordered=True)

        decomposed += mini_batch_size

        new_data.append(i_pca.transform(data))

    return np.concatenate(new_data)
