from sklearn.decomposition import IncrementalPCA
from sklearn.externals import joblib
from helpers import loader_tfidf
from utilities import constants
import pandas as pd
import numpy as np
import random
import json
import os


def reduce(config, uuids, components):
    """
    Apply Incremental Principal Components Analysis to the tf-idf vectors.
    
    :param config: configuration dictionary
    :param uuids: list of selected uuids
    :param components: number of desired components
    :return:
    """

    print('Performing dimensionality reduction using PCA')

    dir_store = config['dir_store']
    core_num = config['core_num']
    mini_batch_size = config['batch_size']

    words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))
    rand_train = random.sample(uuids, len(uuids))

    cols = len(words)
    rows = len(uuids)

    i_pca = IncrementalPCA(n_components=components, batch_size=mini_batch_size)

    train_pca(i_pca, rows, cols, rand_train, words, mini_batch_size, core_num, dir_store)

    print('Explained Variance Ratio')
    print(sum(i_pca.explained_variance_ratio_))

    data = transform_vectors(i_pca, rows, cols, uuids, words, mini_batch_size, core_num, dir_store)

    matrix_file = os.path.join(constants.dir_d, constants.dir_mat, 'pca_{}_{}.txt'.format(components, rows))
    np.savetxt(open(matrix_file, 'wb'), data)

    model_file = os.path.join(constants.dir_d, constants.dir_mat, 'pca_{}_{}.pkl'.format(components, rows))
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
