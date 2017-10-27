from sklearn.decomposition import IncrementalPCA
from sklearn.externals import joblib
from helpers import loader_tfidf
from utilities import constants
import pandas as pd
import numpy as np
import random
import json
import os


def reduce(config, components, uuids=None, x_train=None, x_dev=None, x_test=None):
    """
    Apply Incremental Principal Components Analysis to the tf-idf vectors.
    
    :param config: configuration dictionary
    :param components: number of desired components
    :param uuids: list of selected uuids
    :param x_train: List of train set uuids
    :param x_dev: List of dev set uuids
    :param x_test: List of test set uuids
    :return:
    """

    print('Performing dimensionality reduction using PCA')

    mini_batch_size = config['batch_size']
    words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))

    i_pca = IncrementalPCA(n_components=components, batch_size=mini_batch_size)

    if uuids:
        rand_uuids = random.sample(uuids, len(uuids))
        rows = len(uuids)

    else:
        rand_uuids = random.sample(x_train, len(x_train))
        rows = len(x_train)

    train_pca(config, i_pca, len(rand_uuids), rand_uuids, mini_batch_size)

    print('Explained Variance Ratio {}:'.format(sum(i_pca.explained_variance_ratio_)))

    if uuids:
        data = transform_vectors(config, i_pca, len(uuids), uuids, mini_batch_size)
        matrix_file = os.path.join(constants.dir_d, constants.dir_mat, 'pca_{}_{}.txt'.format(components, len(uuids)))
        np.savetxt(open(matrix_file, 'wb'), data)

    else:
        t_train = transform_vectors(config, i_pca, len(x_train), x_train, mini_batch_size)
        matrix_file = os.path.join(
            constants.dir_d,
            constants.dir_mat,
            'pca_{}_{}_tr.txt'.format(components, len(t_train))
        )
        np.savetxt(open(matrix_file, 'wb'), t_train)

        t_dev = transform_vectors(config, i_pca, len(x_dev), x_dev, mini_batch_size)
        matrix_file = os.path.join(
            constants.dir_d,
            constants.dir_mat,
            'pca_{}_{}_dv.txt'.format(components, len(t_dev))
        )
        np.savetxt(open(matrix_file, 'wb'), t_dev)

        t_test = transform_vectors(config, i_pca, len(x_test), x_test, mini_batch_size)
        matrix_file = os.path.join(
            constants.dir_d,
            constants.dir_mat,
            'pca_{}_{}_te.txt'.format(components, len(t_test))
        )
        np.savetxt(open(matrix_file, 'wb'), t_test)

        data = (t_train, t_dev, t_test)

    model_file = os.path.join(constants.dir_d, constants.dir_mod, 'pca_{}_{}.pkl'.format(components, rows))
    joblib.dump(i_pca, model_file)

    components_file = os.path.join(
        constants.dir_d,
        constants.dir_mod,
        "components_pca_{}_{}.txt".format(components, rows)
    )
    to_inspect = pd.DataFrame(
        np.absolute(i_pca.components_.T),
        index=sorted(set(words.keys())),
        columns=range(components)
    )
    to_inspect.idxmax(axis=0, skipna=True).to_csv(components_file)

    return data, i_pca


def train_pca(config, i_pca, rows, rand_uuids, mini_batch_size):
    """
    Train the PCA algorithm incrementally using mini batches of data.    

    :param config:global configuration dictionary
    :param i_pca: IncrementalPCA object
    :param rows: number of rows of the data set matrix
    :param rand_uuids: list of documents in random order
    :param mini_batch_size: size of each mini batch
    :return:
    """

    decomposed = 0

    while decomposed < rows:
        print('Processing documents from {} to {}'.format(decomposed, (decomposed + mini_batch_size - 1)))
        data = loader_tfidf.load_tfidf(config, rand_uuids[decomposed:][:mini_batch_size], dense=True, ordered=False)

        decomposed += mini_batch_size

        i_pca.partial_fit(data)


def transform_vectors(config, i_pca, rows, uuids, mini_batch_size):
    """
    Transorm the data vectors.

    :param config:global configuration dictionary
    :param i_pca: IncrementalPCA object
    :param rows: number of rows of the data set matrix
    :param uuids: list of documents in order
    :param mini_batch_size: size of each mini batch
    :return: reduced dataset matrix
    :return: 
    """

    decomposed = 0
    new_data = []

    while decomposed < rows:
        print('Transforming documents from {} to {}'.format(decomposed, (decomposed + mini_batch_size - 1)))
        data = loader_tfidf.load_tfidf(config, uuids[decomposed:][:mini_batch_size], dense=True, ordered=True)

        decomposed += mini_batch_size

        new_data.append(i_pca.transform(data))

    return np.concatenate(new_data)
