from sklearn.decomposition import LatentDirichletAllocation
from helpers import loader_freqs
from utilities import constants
import numpy as np
import random
import json
import os


def reduce(config, uuids, components, objective):
    """
    Apply Latent Dirichlet Allocation to the bag of words data-set.

    :param config: configuration dictionary
    :param uuids: list of selected uuids
    :param components: number of desired components
    :return: 
    """

    print('Performing dimensionality reduction using LDA')

    dir_malwords = config['dir_malwords']
    core_num = config['core_num']
    mini_batch_size = config['batch_size']

    words = json.load(open(os.path.join(constants.dir_d, constants.json_words), 'r'))
    rand_uuids = random.sample(uuids, len(uuids))

    cols = len(words)
    rows = len(uuids)

    lda = LatentDirichletAllocation(batch_size=mini_batch_size, n_jobs=core_num, n_topics=components, max_iter=100,
                                    total_samples=rows, learning_method='online', verbose=3)

    train_lda(lda, rows, cols, rand_uuids, words, mini_batch_size, core_num, dir_malwords)

    data = transform_vectors(lda, rows, cols, uuids, words, mini_batch_size, core_num, dir_malwords)

    matrix_file = os.path.join(constants.dir_d, constants.dir_mat, "lda_{}_{}.txt".format(components, rows))
    np.savetxt(open(matrix_file, "wb"), data)

    return data, lda


def train_lda(lda, rows, cols, rand_uuids, words, mini_batch_size, core_num, dir_malwords):
    """
    Train the LDA algorithm incrementally using mini batches of data.    

    :return: 
    """

    decomposed = 0

    while decomposed < rows:
        print('Processing documents from {} to {}'.format(decomposed, (decomposed + mini_batch_size - 1)))
        data = loader_freqs.load_freqs(rand_uuids[decomposed:][:mini_batch_size], core_num, cols, words, dir_malwords,
                                       dense=True, ordered=False)

        decomposed += mini_batch_size

        lda.partial_fit(data)


def transform_vectors(lda, rows, cols, uuids, words, mini_batch_size, core_num, dir_malwords):
    """
    Transorm the data vectors.

    :return: 
    """

    decomposed = 0
    new_data = []

    while decomposed < rows:
        print('Transforming documents from {} to {}'.format(decomposed, (decomposed + mini_batch_size - 1)))
        data = loader_freqs.load_freqs(uuids[decomposed:][:mini_batch_size], core_num, cols, words, dir_malwords,
                                       dense=True, ordered=True)

        decomposed += mini_batch_size

        new_data.append(lda.transform(data))

    return np.concatenate(new_data)
